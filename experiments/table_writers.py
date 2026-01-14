# takes in a set of input tables (each itself in a deltalake/parquet format),
# iterates through them and writes them out using the specified scheme to one or more
# tables
#
#
# ## Strategies
# - **Naive-copy:** Store a single delta table per materialization version, making a new one each time.
# - **Split-seg:** Store a delta table of synapse ID, spatial info. Store a separate one for segmentation
#   info with synapse ID also. Join as needed.
# - **Split-seg-multi:** Store a delta table of synapse ID, spatial info. Store two separate tables, one each for both pre and
#   postsynaptic segmentation (each with synapse IDs). Join as needed.
# - **Delta-columns:** Store a single delta table, append columns per new materialization version as needed.
# - **Delta-rows:** Store a single delta table, but as segmentation changes, update rows. Use delta lake
#   versioning patterns to access prior versions.


# %%

import os
import subprocess
import time
from typing import Union

import polars as pl
from cloudpathlib import AnyPath as Path
from deltalake import DeltaTable, write_deltalake

base_path = Path(
    "/Users/ben.pedigo/code/meshrep/meshrep/data/synapses_pni_2_v1412_deltalake"
)

table = pl.scan_delta(str(base_path))

root_table = pl.read_parquet("./data/column_root_info.parquet")


table.collect_schema()

FIXED_COLS = [
    "id",
    "pre_pt_position_x",
    "pre_pt_position_y",
    "pre_pt_position_z",
    "post_pt_position_x",
    "post_pt_position_y",
    "post_pt_position_z",
    "ctr_pt_position_x",
    "ctr_pt_position_y",
    "ctr_pt_position_z",
    "pre_pt_supervoxel_id",
    "post_pt_supervoxel_id",
    "size",
]

DYNAMIC_COLS = ["pre_pt_root_id", "post_pt_root_id"]

ALL_COLS = FIXED_COLS + DYNAMIC_COLS

# %%


base_version = 1412
currtime = time.time()
example_table = (
    table.rename({"synapse_id": "id"})
    .select(ALL_COLS)
    .filter(
        pl.col("pre_pt_root_id").is_in(root_table["pt_root_id"].to_list())
        # & pl.col("post_pt_root_id").is_in(root_table["pt_root_id"].to_list())
    )
    .collect(engine="streaming")
)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

example_table = (
    example_table.with_columns(
        pl.col("pre_pt_root_id").alias(f"pre_pt_root_id_v{base_version}"),
        pl.col("post_pt_root_id").alias(f"post_pt_root_id_v{base_version}"),
    )
    .drop(["pre_pt_root_id", "post_pt_root_id"])
    .lazy()
)

# %%
other_versions = [343, 1507]

for version in other_versions:
    other_seg_cols = pl.scan_parquet(
        f"/Users/ben.pedigo/code/parqlayground/parqlayground/data/synapses_pni_2_v1_filtered_view_v{version}_segmentation_cols.parquet"
    )
    example_table = example_table.join(
        other_seg_cols.select(
            "id",
            pl.col("pre_pt_root_id").alias(f"pre_pt_root_id_v{version}"),
            pl.col("post_pt_root_id").alias(f"post_pt_root_id_v{version}"),
        ),
        on="id",
        how="left",
    )
currtime = time.time()
example_table = example_table.collect(engine="streaming")
print(f"{time.time() - currtime:.3f} seconds elapsed.")

# %%
# now, example_table is in "delta-columns" format.
# let's write code to get it into the other formats. note that it's not in memory yet.


def write_table(table: Union[pl.LazyFrame, pl.DataFrame], out_path: Union[str, Path]):
    if isinstance(table, pl.LazyFrame):
        table = table.collect(engine="streaming")
    write_deltalake(str(out_path), table, mode="append")


def upsert_table(
    new_table: Union[pl.LazyFrame, pl.DataFrame], target_table: Union[str, Path]
) -> int:
    if isinstance(new_table, pl.LazyFrame):
        new_table = new_table.collect(engine="streaming")
    dt = DeltaTable(str(target_table))
    (
        dt.merge(
            source=new_table,
            predicate="target.id = source.id",
            source_alias="source",
            target_alias="target",
        )
        .when_matched_update_all()
        .execute()
    )
    dt.create_checkpoint()
    return dt.version()


base_out_path = Path(
    "/Users/ben.pedigo/code/parqlayground/parqlayground/data/write_out_test"
)
# empty out directory


subprocess.run(["rm", "-rf", str(base_out_path)])
subprocess.run(["mkdir", "-p", str(base_out_path)])

versions = [base_version] + other_versions
versions = sorted(versions)

# Naive-copy format
for version in versions:
    out_table = example_table.select(
        FIXED_COLS
        + [
            pl.col(f"pre_pt_root_id_v{version}").alias("pre_pt_root_id"),
            pl.col(f"post_pt_root_id_v{version}").alias("post_pt_root_id"),
        ]
    )
    out_path = base_out_path / f"naive_copy_v{version}"  # ty: ignore
    write_table(out_table, out_path)

# Split-seg format
fixed_table = example_table.select(FIXED_COLS)
out_path = base_out_path / "fixed_table"  # ty: ignore
write_table(fixed_table, out_path)

for version in versions:
    version_dynamic_cols = ["id"] + [
        f"pre_pt_root_id_v{version}",
        f"post_pt_root_id_v{version}",
    ]
    dynamic_table = example_table.select(version_dynamic_cols).rename(
        {
            f"pre_pt_root_id_v{version}": "pre_pt_root_id",
            f"post_pt_root_id_v{version}": "post_pt_root_id",
        }
    )
    out_path = base_out_path / f"dynamic_table_v{version}"  # ty: ignore
    write_table(dynamic_table, out_path)

# Split-seg-multi format
# Note: uses the same fixed_table as above
for version in versions:
    pre_dynamic_table = example_table.select(
        ["id", f"pre_pt_root_id_v{version}"]
    ).rename({f"pre_pt_root_id_v{version}": "pre_pt_root_id"})
    out_path = base_out_path / f"pre_dynamic_table_v{version}"  # ty: ignore
    write_table(pre_dynamic_table, out_path)

    post_dynamic_table = example_table.select(
        ["id", f"post_pt_root_id_v{version}"]
    ).rename({f"post_pt_root_id_v{version}": "post_pt_root_id"})
    out_path = base_out_path / f"post_dynamic_table_v{version}"  # ty: ignore
    write_table(post_dynamic_table, out_path)

# Delta-columns format (already done, just write out)
out_path = base_out_path / "delta_columns"  # ty: ignore
write_table(example_table, out_path)

# Delta-rows format
# write out the first version
version = versions[0]
out_table = example_table.select(
    FIXED_COLS
    + [
        pl.col(f"pre_pt_root_id_v{version}").alias("pre_pt_root_id"),
        pl.col(f"post_pt_root_id_v{version}").alias("post_pt_root_id"),
    ]
)
out_path = base_out_path / "delta_rows"  # ty: ignore
write_table(out_table, out_path)
delta_rows_version_map = {version: 0}
for last_version, current_version in zip(versions[:-1], versions[1:]):
    # create a table with updates from the current version
    update_table = (
        example_table.filter(
            (
                pl.col(f"pre_pt_root_id_v{current_version}")
                != pl.col(f"pre_pt_root_id_v{last_version}")
            )
            | (
                pl.col(f"post_pt_root_id_v{current_version}")
                != pl.col(f"post_pt_root_id_v{last_version}")
            )
        )
        .rename(
            {
                f"pre_pt_root_id_v{current_version}": "pre_pt_root_id",
                f"post_pt_root_id_v{current_version}": "post_pt_root_id",
            }
        )
        .select(ALL_COLS)
    )
    print(len(update_table))
    delta_version = upsert_table(update_table, out_path)
    delta_rows_version_map[current_version] = delta_version

    # TODO: may need to index the materialization version to the delta lake version

# Split-seg-delta-rows format
# Note: uses the same fixed_table as above
version = versions[0]
out_table = example_table.select(
    [
        pl.col("id"),
        pl.col(f"pre_pt_root_id_v{version}").alias("pre_pt_root_id"),
        pl.col(f"post_pt_root_id_v{version}").alias("post_pt_root_id"),
    ]
)
out_path = base_out_path / "split_seg_delta_rows"  # ty: ignore
write_table(out_table, out_path)
split_seg_delta_version_map = {version: 0}
for last_version, current_version in zip(versions[:-1], versions[1:]):
    # create a table with updates from the current version
    update_table = (
        example_table.filter(
            (
                pl.col(f"pre_pt_root_id_v{current_version}")
                != pl.col(f"pre_pt_root_id_v{last_version}")
            )
            | (
                pl.col(f"post_pt_root_id_v{current_version}")
                != pl.col(f"post_pt_root_id_v{last_version}")
            )
        )
        .rename(
            {
                f"pre_pt_root_id_v{current_version}": "pre_pt_root_id",
                f"post_pt_root_id_v{current_version}": "post_pt_root_id",
            }
        )
        .select(["id", "pre_pt_root_id", "post_pt_root_id"])
    )
    print(len(update_table))
    delta_version = upsert_table(update_table, out_path)
    split_seg_delta_version_map[current_version] = delta_version

# %%
# tracking storage per format


def get_dir_size(path: Union[str, Path]) -> int:
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


# %%

dirs_by_method = {
    "naive_copy": [f"naive_copy_v{v}" for v in versions],
    "split_seg": ["fixed_table"] + [f"dynamic_table_v{v}" for v in versions],
    "split_seg_multi": ["fixed_table"]
    + [f"pre_dynamic_table_v{v}" for v in versions]
    + [f"post_dynamic_table_v{v}" for v in versions],
    "delta_columns": ["delta_columns"],
    "delta_rows": ["delta_rows"],
    "split_seg_delta_rows": ["fixed_table"] + ["split_seg_delta_rows"],
}

for method, dir_names in dirs_by_method.items():
    total_size = 0
    for dir_name in dir_names:
        dir_path = base_out_path / dir_name  # ty: ignore
        dir_size = get_dir_size(dir_path)
        total_size += dir_size
    print(f"{method}: {total_size / (1000**3):.3f} GiB")

# %%

pl.read_delta(
    "/Users/ben.pedigo/code/parqlayground/parqlayground/data/write_out_test/delta_rows",
    version=2,
).filter(pl.col("id") == 203270208).select("post_pt_root_id")

# %%

# reconstruct each of the above as lazy frames - some will require joins on scanned delta tables


def scan_naive_copy(version: int) -> pl.LazyFrame:
    path = base_out_path / f"naive_copy_v{version}"  # ty: ignore
    out = pl.scan_delta(str(path)).select(ALL_COLS)
    return out


scan_naive_copy(1507).collect_schema()


def scan_split_seg(version: int) -> pl.LazyFrame:
    fixed_path = base_out_path / "fixed_table"  # ty: ignore
    dynamic_path = base_out_path / f"dynamic_table_v{version}"  # ty: ignore
    fixed_table = pl.scan_delta(str(fixed_path))
    dynamic_table = pl.scan_delta(str(dynamic_path))
    out = fixed_table.join(dynamic_table, on="id", how="inner").select(ALL_COLS)
    return out


scan_split_seg(1507).collect_schema()


def scan_split_seg_multi(version: int) -> pl.LazyFrame:
    fixed_path = base_out_path / "fixed_table"  # ty: ignore
    pre_dynamic_path = base_out_path / f"pre_dynamic_table_v{version}"  # ty: ignore
    post_dynamic_path = base_out_path / f"post_dynamic_table_v{version}"  # ty: ignore
    fixed_table = pl.scan_delta(str(fixed_path))
    pre_dynamic_table = pl.scan_delta(str(pre_dynamic_path))
    post_dynamic_table = pl.scan_delta(str(post_dynamic_path))
    out = (
        fixed_table.join(pre_dynamic_table, on="id", how="inner")
        .join(post_dynamic_table, on="id", how="inner")
        .select(ALL_COLS)
    )
    return out


scan_split_seg_multi(1507).collect_schema()


def scan_delta_columns(version: int) -> pl.LazyFrame:
    path = base_out_path / "delta_columns"  # ty: ignore
    out = (
        pl.scan_delta(str(path))
        .select(
            FIXED_COLS + [f"pre_pt_root_id_v{version}", f"post_pt_root_id_v{version}"]
        )
        .rename(
            {
                f"pre_pt_root_id_v{version}": "pre_pt_root_id",
                f"post_pt_root_id_v{version}": "post_pt_root_id",
            }
        )
    )
    return out


scan_delta_columns(1507).collect_schema()


def scan_delta_rows(version: int) -> pl.LazyFrame:
    path = base_out_path / "delta_rows"  # ty: ignore
    delta_version_number = delta_rows_version_map[version]
    out = pl.scan_delta(str(path), version=delta_version_number).select(ALL_COLS)
    return out


scan_delta_rows(1507).collect_schema()


def scan_split_seg_delta_rows(version: int) -> pl.LazyFrame:
    fixed_path = base_out_path / "fixed_table"  # ty: ignore
    dynamic_path = base_out_path / "split_seg_delta_rows"  # ty: ignore
    fixed_table = pl.scan_delta(str(fixed_path))
    delta_version_number = split_seg_delta_version_map[version]
    dynamic_table = pl.scan_delta(str(dynamic_path), version=delta_version_number)
    out = fixed_table.join(dynamic_table, on="id", how="inner").select(ALL_COLS)
    return out


scan_split_seg_delta_rows(1507).collect_schema()

# %%
dt = DeltaTable(str(base_out_path / "delta_rows"))

dt.history()
