#!/usr/bin/env python3
"""
Convert materialized table data to DeltaLake format.

This script downloads materialized table data from cloud storage,
processes it in chunks, and converts it to DeltaLake format with
optimizations like partitioning, Z-order sorting, and bloom filters.
"""

import subprocess
import time
from functools import partial
from typing import List, Optional

import numpy as np
import pandas as pd
import polars as pl
import typer
from caveclient import CAVEclient
from cloudpathlib import AnyPath as Path
from cloudvolume import CloudVolume
from deltalake import DeltaTable, write_deltalake
from deltalake.table import TableOptimizer
from deltalake.writer import BloomFilterProperties, ColumnProperties, WriterProperties
from shapely import wkb

# Constants and helper functions
SQL_TO_POLARS_DTYPE = {
    "bigint": pl.Int64,
    "integer": pl.Int32,
    "smallint": pl.Int16,
    "real": pl.Float32,
    "double precision": pl.Float64,
    "numeric": pl.Decimal,
    "boolean": pl.Boolean,
    "text": pl.String,
    "varchar": pl.String,
    "character varying": pl.String,
    "date": pl.Date,
    "timestamp without time zone": pl.Datetime,
    "timestamp with time zone": pl.Datetime,
}


def sql_to_polars_dtype(sql_type: str) -> pl.datatypes.DataType:
    """
    Convert a SQL dtype string to a Polars dtype.
    Raises ValueError if the dtype is not recognized.
    """
    sql_type = sql_type.strip().lower()
    # handle e.g. 'character varying(255)'
    if "(" in sql_type:
        sql_type = sql_type.split("(")[0].strip()
    if sql_type not in SQL_TO_POLARS_DTYPE:
        valid = ", ".join(sorted(SQL_TO_POLARS_DTYPE))
        raise ValueError(
            f"Unrecognized SQL dtype: {sql_type!r}. Valid options: {valid}"
        )
    return SQL_TO_POLARS_DTYPE[sql_type]


def build_polars_schema(schema_df):
    """
    Given a DataFrame with columns ['field', 'dtype'],
    return a dict usable as a Polars schema.
    """
    return {
        row.field: sql_to_polars_dtype(row.dtype)
        for row in schema_df.itertuples(index=False)
    }


def decoder(x: str) -> np.ndarray:
    """
    Decoder for WKB point columns, if necessary.
    TODO: int32 is hard-coded here, make more flexible
    """
    point = wkb.loads(bytes.fromhex(x))
    out = np.array([point.x, point.y, point.z], dtype=np.int32)
    return out


def id_partition_func(
    id_to_encode: int,
    n_partitions: int = 256,
    use_seg_id: bool = False,
    cv: Optional[CloudVolume] = None,
) -> np.uint16:
    """
    Function to determine partition for an ID.
    """
    if id_to_encode == 0:
        return np.uint16(0)
    if use_seg_id:
        id_to_encode = cv.meta.decode_segid(id_to_encode)

    # salt = 123456
    # partition = hash(id_to_encode ^ salt) % n_partitions
    # partition = ((id_to_encode * 2654435761) & 0xFFFFFFFF) % n_partitions

    partition = id_to_encode % n_partitions
    return np.uint16(partition)


DATASTACK_TO_CLOUD_PATH = {"minnie65_phase3_v1": "gs://mat_dbs/public/"}

BASE_OUT_PATH = "gs://allen-minnie-phase3/bdp-deltalakes"

DYNAMIC_COLS = ["id", "pre_pt_root_id", "post_pt_root_id"]


def main(
    datastack: str = typer.Argument(..., help="Name of the datastack"),
    table_name: str = typer.Argument(..., help="Name of the table or view to process"),
    version: int = typer.Argument(..., help="Materialization version"),
    base_out_path: str = typer.Option(
        BASE_OUT_PATH, help="Output path for DeltaLake table"
    ),
    drop_columns: Optional[List[str]] = typer.Option(
        ["created", "deleted", "superceded_id", "valid"],
        "--drop-column",
        help="Columns to drop from the table (can be used multiple times)",
    ),
    n_rows_per_chunk: int = typer.Option(
        50_000_000, "--chunk-size", help="Number of rows to process per chunk"
    ),
    partition_column: str = typer.Option(
        "post_pt_root_id", "--partition-column", help="Column to use for partitioning"
    ),
    n_partitions: int = typer.Option(
        64, "--partitions", help="Number of partitions to create"
    ),
    use_seg_id: bool = typer.Option(
        False,
        "--use-seg-id",
        help="Whether to convert IDs to segmentation IDs before partitioning",
    ),
    zorder_columns: Optional[List[str]] = typer.Option(
        ["post_pt_root_id", "id"],
        "--zorder-column",
        help="Z-order curve columns for optimization (can be used multiple times)",
    ),
    bloom_filter_columns: Optional[List[str]] = typer.Option(
        ["id"],
        "--bloom-filter-column",
        help="Columns to add bloom filters to (can be used multiple times)",
    ),
    fpp: float = typer.Option(
        0.001, "--fpp", help="False positive probability for bloom filters"
    ),
) -> None:
    """Convert materialized table data to DeltaLake format."""

    total_time = time.time()

    # Handle empty lists from typer
    drop_columns = drop_columns or []
    zorder_columns = zorder_columns or []
    bloom_filter_columns = bloom_filter_columns or []

    mat_db_cloud_path = DATASTACK_TO_CLOUD_PATH[datastack]
    base_cloud_path = Path(f"{mat_db_cloud_path}/{datastack}/v{version}")
    table_file_name = f"{table_name}.csv.gz"
    header_file_name = f"{table_name}_header.csv"
    table_cloud_path = base_cloud_path / table_file_name
    header_cloud_path = base_cloud_path / header_file_name

    typer.echo("Working on table:")
    typer.echo(str(table_cloud_path))
    typer.echo(str(header_cloud_path))
    typer.echo()

    out_path = Path(f"{base_out_path}/{datastack}/v{version}/{table_name}")
    typer.echo(f"Output DeltaLake path: {out_path}")
    typer.echo()

    # check the file sizes and that they exist
    typer.echo(f"Table size: {table_cloud_path.stat().st_size / 1e9:.3f} GB")
    typer.echo(f"Header size: {header_cloud_path.stat().st_size / 1e3:.3f} KB")
    typer.echo()

    # download the table and header files locally
    download_time = time.time()

    typer.echo("Downloading table and header files...")

    temp_path = Path("/tmp/table_to_deltalake")
    temp_path.mkdir(exist_ok=True)
    table_local_path = temp_path / table_file_name
    header_local_path = temp_path / header_file_name

    downloaded = True
    if not downloaded:
        subprocess.run(
            [
                "gsutil",
                "cp",
                str(table_cloud_path),
                str(temp_path / table_cloud_path.name),
            ]
        )
        subprocess.run(
            [
                "gsutil",
                "cp",
                str(header_cloud_path),
                str(temp_path / header_cloud_path.name),
            ]
        )

        typer.echo(
            f"{time.time() - download_time:.3f} seconds elapsed to download files."
        )
        typer.echo()

        # unzip the table
        # this was more reliable for large files than using pandas/polars unzip directly for me

        unzip_time = time.time()

        typer.echo("Unzipping table file...")

        subprocess.run(
            [
                "gunzip",
                str(table_local_path),
            ]
        )

        typer.echo(f"{time.time() - unzip_time:.3f} seconds elapsed to unzip table.")
        typer.echo()

    table_local_path = temp_path / f"{table_name}.csv"

    # create a plan for reading in data via polars

    write_time = time.time()

    typer.echo("Reading in table and writing to deltalake...")

    header = pd.read_csv(header_local_path, header=None).rename(
        columns={0: "field", 1: "dtype"}
    )

    schema = build_polars_schema(header)

    table = pl.scan_csv(table_local_path, has_header=False, schema=schema).drop(
        drop_columns, strict=False
    )

    schema = table.collect_schema()

    typer.echo("Reading in table with schema:")
    for key, val in schema.items():
        typer.echo(f"{key}: {val}")
    typer.echo()

    columns = table.collect_schema().names()

    # intended to only catch unpacked point columns
    position_columns = [c for c in columns if (c.endswith("_pt_position"))]

    if len(position_columns) > 0:
        typer.echo(f"Decoding {len(position_columns)} position columns...")
        table = table.with_columns(
            pl.col(position_columns).map_elements(
                decoder, return_dtype=pl.List(pl.Int32)
            )
        )

    if use_seg_id:
        client = CAVEclient(datastack, version=version)
        cv = client.info.segmentation_cloudvolume()
    else:
        cv = None

    partial_id_partition_func = partial(
        id_partition_func,
        n_partitions=n_partitions,
        use_seg_id=use_seg_id,
        cv=cv,
    )

    partition_by = f"{partition_column}_partition"

    table = table.with_columns(
        pl.col(partition_column)
        .map_elements(
            partial_id_partition_func,
            return_dtype=pl.UInt16,
        )
        .alias(partition_by)
    )

    write_mode = "append"
    unfinished = True

    start = 0
    while unfinished:
        typer.echo(
            f"Processing chunk for rows {start:,} to {start + n_rows_per_chunk:,}..."
        )
        chunk_table = table.slice(start, n_rows_per_chunk).collect()

        # Process the chunk...
        # If the chunk is empty, we're done
        if chunk_table.is_empty():
            unfinished = False
        else:
            start += n_rows_per_chunk

        write_deltalake(
            out_path.as_uri(),
            chunk_table,
            partition_by=partition_by,
            mode=write_mode,
        )

    typer.echo(
        f"{time.time() - write_time:.3f} seconds elapsed to read and write table."
    )
    typer.echo()

    # delete the temporary files

    delete_time = time.time()

    typer.echo("Cleaning up temporary files...")
    subprocess.run(["rm", str(table_local_path)])
    subprocess.run(["rm", str(header_local_path)])
    typer.echo(
        f"{time.time() - delete_time:.3f} seconds elapsed to delete temporary files."
    )
    typer.echo()

    # optimize the deltalake with z-ordering and bloom filters

    optimize_time = time.time()

    typer.echo("Optimizing deltalake...")
    if len(bloom_filter_columns) > 0:
        bloom = BloomFilterProperties(
            set_bloom_filter_enabled=True,
            fpp=fpp,
        )
        column_properties = ColumnProperties(bloom_filter_properties=bloom)
        writer_properties = WriterProperties(
            column_properties={col: column_properties for col in bloom_filter_columns}
        )
    else:
        writer_properties = None

    dt = DeltaTable(out_path.as_uri())
    to = TableOptimizer(dt)
    to.z_order(columns=zorder_columns, writer_properties=writer_properties)
    dt.vacuum(
        dry_run=False, retention_hours=0, enforce_retention_duration=False, full=True
    )

    typer.echo(
        f"{time.time() - optimize_time:.3f} seconds elapsed to optimize deltalake."
    )
    typer.echo()

    typer.echo("Done!")
    typer.echo("-----------------")
    typer.echo(f"{time.time() - total_time:.3f} seconds elapsed total.")
    typer.echo("-----------------")


if __name__ == "__main__":
    typer.run(main)
