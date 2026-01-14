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

import subprocess
import time
from pathlib import Path as LocalPath
from typing import Union

import polars as pl
from cloudpathlib import AnyPath as Path
from deltalake import (
    DeltaTable,
    WriterProperties,
    write_deltalake,
)
from deltalake.table import TableOptimizer  # noqa
from tqdm.auto import tqdm

base_path = Path(
    "/Users/ben.pedigo/code/meshrep/meshrep/data/synapses_pni_2_v1412_deltalake"
)

root_table = pl.read_parquet("./data/column_root_info.parquet")

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

table = pl.scan_delta(str(base_path))

base_version = 1412
currtime = time.time()
table = (
    table.rename({"synapse_id": "id"})
    .select(ALL_COLS)
    .slice(0, 20_000_000)
    # .filter(
    #     pl.col("pre_pt_root_id").is_in(root_table["pt_root_id"].to_list())
    #     | pl.col("post_pt_root_id").is_in(root_table["pt_root_id"].to_list())
    # )
    .collect(engine="streaming")
)
print(f"{time.time() - currtime:.3f} seconds elapsed.")
print(len(table))

table = table.rename(
    {
        "pre_pt_root_id": f"pre_pt_root_id_v{base_version}",
        "post_pt_root_id": f"post_pt_root_id_v{base_version}",
    }
).lazy()


# %%
other_versions = [343, 1507]

for version in other_versions:
    other_seg_cols = pl.scan_parquet(
        f"/Users/ben.pedigo/code/parqlayground/parqlayground/data/synapses_pni_2_v1_filtered_view_v{version}_segmentation_cols.parquet"
    )
    table = table.join(
        other_seg_cols.select(
            "id",
            pl.col("pre_pt_root_id").alias(f"pre_pt_root_id_v{version}"),
            pl.col("post_pt_root_id").alias(f"post_pt_root_id_v{version}"),
        ),
        on="id",
        how="left",
    )
currtime = time.time()
table = table.collect(engine="streaming")
print(f"{time.time() - currtime:.3f} seconds elapsed.")

# %%
# now, table is in "delta-columns" format.
# let's write code to get it into the other formats. note that it's not in memory yet.


wp = WriterProperties(
    # compression="ZSTD",
    compression=None,
    # compression_level=3,
)
# partition_by = ["id_partition"]


def write_table(table: Union[pl.LazyFrame, pl.DataFrame], out_path: Union[str, Path]):
    if isinstance(table, pl.LazyFrame):
        table = table.collect(engine="streaming")

    # table = table.with_columns((pl.col("id") // 64).alias("id_partition"))

    write_deltalake(
        str(out_path),
        table,
        mode="append",
        writer_properties=wp,
        # partition_by=partition_by,
    )


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
            writer_properties=wp,
        )
        # .when_matched_update_all()
        .when_matched_update(
            updates={
                "pre_pt_root_id": "source.pre_pt_root_id",
                "post_pt_root_id": "source.post_pt_root_id",
            },
            predicate="target.id = source.id",
        )
        # .when_matched_delete("target.id = source.id")
        .execute()
    )
    # dt.update
    # write_deltalake(str(out_path), new_table, mode="append", writer_properties=wp)

    return dt.version()


base_out_path = Path(
    "/Users/ben.pedigo/code/parqlayground/parqlayground/data/write_out_test"
)
# empty out directory
subprocess.run(["rm", "-rf", str(base_out_path)])
subprocess.run(["mkdir", "-p", str(base_out_path)])

versions = [base_version] + other_versions
versions = sorted(versions)

# -----------------
# Naive-copy format
for version in versions:
    out_table = table.select(
        FIXED_COLS
        + [
            pl.col(f"pre_pt_root_id_v{version}").alias("pre_pt_root_id"),
            pl.col(f"post_pt_root_id_v{version}").alias("post_pt_root_id"),
        ]
    )
    out_path = base_out_path / f"naive_copy_v{version}"  # ty: ignore
    write_table(out_table, out_path)

# ----------------
# Split-seg format
fixed_table = table.select(FIXED_COLS)
out_path = base_out_path / "fixed_table"  # ty: ignore
write_table(fixed_table, out_path)

for version in versions:
    version_dynamic_cols = ["id"] + [
        f"pre_pt_root_id_v{version}",
        f"post_pt_root_id_v{version}",
    ]
    dynamic_table = table.select(version_dynamic_cols).rename(
        {
            f"pre_pt_root_id_v{version}": "pre_pt_root_id",
            f"post_pt_root_id_v{version}": "post_pt_root_id",
        }
    )
    out_path = base_out_path / f"dynamic_table_v{version}"  # ty: ignore
    write_table(dynamic_table, out_path)

# ----------------------
# Split-seg-multi format
# Note: uses the same fixed_table as above
for version in versions:
    pre_dynamic_table = table.select(["id", f"pre_pt_root_id_v{version}"]).rename(
        {f"pre_pt_root_id_v{version}": "pre_pt_root_id"}
    )
    out_path = base_out_path / f"pre_dynamic_table_v{version}"  # ty: ignore
    write_table(pre_dynamic_table, out_path)

    post_dynamic_table = table.select(["id", f"post_pt_root_id_v{version}"]).rename(
        {f"post_pt_root_id_v{version}": "post_pt_root_id"}
    )
    out_path = base_out_path / f"post_dynamic_table_v{version}"  # ty: ignore
    write_table(post_dynamic_table, out_path)

# ---------------------------------------------------
# Delta-columns format (already done, just write out)
out_path = base_out_path / "delta_columns"  # ty: ignore
write_table(table, out_path)

# -----------------
# Delta-rows format
# write out the first version
version = versions[0]
out_table = table.select(
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
        table.filter(
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

# -----------
# dumb_delta_rows - just add a new column "version" to identify version for each row

# dt = DeltaTable(str(out_path))
# to = TableOptimizer(dt)
# to.compact()
# to.z_order(columns=['id'])
# dt.vacuum(
#     dry_run=False,
#     retention_hours=0,
#     enforce_retention_duration=False,
#     full=True,
#     keep_versions=list(delta_rows_version_map.values()),
# )

# ---------------------------
# Split-seg-delta-rows format
# Note: uses the same fixed_table as above
version = versions[0]
out_table = table.select(
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
        table.filter(
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


def get_dir_size(path: Union[str, LocalPath]) -> int:
    total_size = 0
    for file in LocalPath(path).rglob("*"):
        if file.is_file():
            total_size += file.stat().st_size
    return total_size


size_rows = []
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
    print(f"{method}: {total_size / (1000**3):.3f} gb")

    size_rows.append({"method": method, "size_gb": total_size / (1000**3)})

size_results = pl.DataFrame(size_rows)
# %%

pl.read_delta(
    "/Users/ben.pedigo/code/parqlayground/parqlayground/data/write_out_test/delta_rows",
    version=1,
).filter(pl.col("id") == 203270208).select("post_pt_root_id")

# %%

dt = DeltaTable(
    "/Users/ben.pedigo/code/parqlayground/parqlayground/data/write_out_test/delta_rows"
)
dt.history()

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


def query_single_synapse_by_id(table: pl.LazyFrame, synapse_id: int) -> pl.LazyFrame:
    return table.filter(pl.col("id") == synapse_id)


def query_many_synapses_by_ids(
    table: pl.LazyFrame, synapse_ids: list[int]
) -> pl.LazyFrame:
    return table.filter(pl.col("id").is_in(synapse_ids))


def query_post_synapses_by_root_id(table: pl.LazyFrame, root_id: int) -> pl.LazyFrame:
    return table.filter(pl.col("post_pt_root_id") == root_id)


def query_pre_synapses_by_root_id(table: pl.LazyFrame, root_id: int) -> pl.LazyFrame:
    return table.filter(pl.col("pre_pt_root_id") == root_id)


def query_post_synapses_mean_size_by_root_id(
    table: pl.LazyFrame, root_id: int
) -> pl.LazyFrame:
    return (
        table.filter(pl.col("post_pt_root_id") == root_id).select(pl.col("size")).mean()
    )


def query_pre_synapses_mean_size_by_root_id(
    table: pl.LazyFrame, root_id: int
) -> pl.LazyFrame:
    return (
        table.filter(pl.col("pre_pt_root_id") == root_id).select(pl.col("size")).mean()
    )


def query_post_synapses_by_root_id_group(
    table: pl.LazyFrame, root_ids: list[int]
) -> pl.LazyFrame:
    return table.filter(pl.col("post_pt_root_id").is_in(root_ids))


def query_pre_synapses_by_root_id_group(
    table: pl.LazyFrame, root_ids: list[int]
) -> pl.LazyFrame:
    return table.filter(pl.col("pre_pt_root_id").is_in(root_ids))


def query_induced_synapses_by_root_id_group(
    table: pl.LazyFrame, root_ids: list[int]
) -> pl.LazyFrame:
    return table.filter(
        pl.col("pre_pt_root_id").is_in(root_ids)
        & pl.col("post_pt_root_id").is_in(root_ids)
    )


def query_induced_connections_by_root_id_group(
    table: pl.LazyFrame, root_ids: list[int]
) -> pl.LazyFrame:
    return (
        table.filter(
            pl.col("pre_pt_root_id").is_in(root_ids)
            & pl.col("post_pt_root_id").is_in(root_ids)
        )
        .group_by(["pre_pt_root_id", "post_pt_root_id"])
        .agg(
            pl.len().alias("n_synapses"),
            pl.col("size").mean().alias("mean_synapse_size"),
        )
    )


# %%

base_table = table.with_columns(
    pl.col(f"post_pt_root_id_v{base_version}").alias("post_pt_root_id"),
    pl.col(f"pre_pt_root_id_v{base_version}").alias("pre_pt_root_id"),
)


seed = 8888

method_scanners = {
    "naive_copy": scan_naive_copy,
    "split_seg": scan_split_seg,
    "split_seg_multi": scan_split_seg_multi,
    "delta_columns": scan_delta_columns,
    "delta_rows": scan_delta_rows,
    "split_seg_delta_rows": scan_split_seg_delta_rows,
}

rows = []

# query single synapses by id
n_trials = 10
select_synapse_ids = base_table.select("id").sample(n_trials, seed=seed).to_series()
for method, scanner in tqdm(
    method_scanners.items(), total=len(method_scanners), desc="single_synapse_by_id"
):
    for version in versions:
        scanned_version_table = scanner(version)
        for i, syn_id in enumerate(select_synapse_ids):
            currtime = time.time()
            query = query_single_synapse_by_id(scanned_version_table, syn_id)
            result = query.collect()
            dt = time.time() - currtime
            rows.append(
                {
                    "method": method,
                    "version": version,
                    "query_id": i,
                    "n_result_rows": len(result),
                    "query_time_sec": dt,
                    "task": "single_synapse_by_id",
                }
            )

# query multiple_synapses_by_ids
n_trials = 10
n_synapses_per_query = 100
select_synapse_sets = [
    base_table.select("id").sample(n_synapses_per_query).to_series().to_list()
    for _ in range(n_trials)
]
for method, scanner in tqdm(
    method_scanners.items(), total=len(method_scanners), desc="many_synapses_by_ids"
):
    for version in versions:
        scanned_version_table = scanner(version)
        for i, synapse_ids in enumerate(select_synapse_sets):
            currtime = time.time()
            query = query_many_synapses_by_ids(scanned_version_table, synapse_ids)
            result = query.collect()
            dt = time.time() - currtime
            rows.append(
                {
                    "method": method,
                    "version": version,
                    "query_id": i,
                    "n_result_rows": len(result),
                    "query_time_sec": dt,
                    "task": "many_synapses_by_ids",
                }
            )

# query post synapses by root id
n_trials = 10
select_root_ids = (
    base_table.select("post_pt_root_id").unique().sample(n_trials).to_series()
)
for method, scanner in tqdm(
    method_scanners.items(), total=len(method_scanners), desc="post_synapses_by_root_id"
):
    for version in versions:
        scanned_version_table = scanner(version)
        for i, root_id in enumerate(select_root_ids):
            currtime = time.time()
            query = query_post_synapses_by_root_id(scanned_version_table, root_id)
            result = query.collect()
            dt = time.time() - currtime
            rows.append(
                {
                    "method": method,
                    "version": version,
                    "query_id": i,
                    "n_result_rows": len(result),
                    "query_time_sec": dt,
                    "task": "post_synapses_by_root_id",
                }
            )

# query pre synapses by root id
n_trials = 10
select_root_ids = (
    base_table.select("pre_pt_root_id").unique().sample(n_trials).to_series()
)
for method, scanner in tqdm(
    method_scanners.items(), total=len(method_scanners), desc="pre_synapses_by_root_id"
):
    for version in versions:
        scanned_version_table = scanner(version)
        for i, root_id in enumerate(select_root_ids):
            currtime = time.time()
            query = query_pre_synapses_by_root_id(scanned_version_table, root_id)
            result = query.collect()
            dt = time.time() - currtime
            rows.append(
                {
                    "method": method,
                    "version": version,
                    "query_id": i,
                    "n_result_rows": len(result),
                    "query_time_sec": dt,
                    "task": "pre_synapses_by_root_id",
                }
            )

# query post synapses mean size by root_id
n_trials = 10
select_root_ids = (
    base_table.select("post_pt_root_id").unique().sample(n_trials).to_series()
)
for method, scanner in tqdm(
    method_scanners.items(),
    total=len(method_scanners),
    desc="post_synapses_mean_size_by_root_id",
):
    for version in versions:
        scanned_version_table = scanner(version)
        for i, root_id in enumerate(select_root_ids):
            currtime = time.time()
            query = query_post_synapses_mean_size_by_root_id(
                scanned_version_table, root_id
            )
            result = query.collect()
            dt = time.time() - currtime
            rows.append(
                {
                    "method": method,
                    "version": version,
                    "query_id": i,
                    "n_result_rows": len(result),
                    "query_time_sec": dt,
                    "task": "post_synapses_mean_size_by_root_id",
                }
            )

# query pre synapses mean size by root_id
n_trials = 10
select_root_ids = (
    base_table.select("pre_pt_root_id").unique().sample(n_trials).to_series()
)
for method, scanner in tqdm(
    method_scanners.items(),
    total=len(method_scanners),
    desc="pre_synapses_mean_size_by_root_id",
):
    for version in versions:
        scanned_version_table = scanner(version)
        for i, root_id in enumerate(select_root_ids):
            currtime = time.time()
            query = query_pre_synapses_mean_size_by_root_id(
                scanned_version_table, root_id
            )
            result = query.collect()
            dt = time.time() - currtime
            rows.append(
                {
                    "method": method,
                    "version": version,
                    "query_id": i,
                    "n_result_rows": len(result),
                    "query_time_sec": dt,
                    "task": "pre_synapses_mean_size_by_root_id",
                }
            )

# query post synapses by root id group
n_trials = 10
n_roots_per_query = 100
select_root_id_sets = [
    base_table.select("post_pt_root_id")
    .unique()
    .sample(n_roots_per_query)
    .to_series()
    .to_list()
    for _ in range(n_trials)
]
for method, scanner in tqdm(
    method_scanners.items(),
    total=len(method_scanners),
    desc="post_synapses_by_root_id_group",
):
    for version in versions:
        scanned_version_table = scanner(version)
        for i, root_ids in enumerate(select_root_id_sets):
            currtime = time.time()
            query = query_post_synapses_by_root_id_group(
                scanned_version_table, root_ids
            )
            result = query.collect()
            dt = time.time() - currtime
            rows.append(
                {
                    "method": method,
                    "version": version,
                    "query_id": i,
                    "n_result_rows": len(result),
                    "query_time_sec": dt,
                    "task": "post_synapses_by_root_id_group",
                }
            )

# query pre synapses by root id group
n_trials = 10
n_roots_per_query = 100
select_root_id_sets = [
    base_table.select("pre_pt_root_id")
    .unique()
    .sample(n_roots_per_query)
    .to_series()
    .to_list()
    for _ in range(n_trials)
]
for method, scanner in tqdm(
    method_scanners.items(),
    total=len(method_scanners),
    desc="pre_synapses_by_root_id_group",
):
    for version in versions:
        scanned_version_table = scanner(version)
        for i, root_ids in enumerate(select_root_id_sets):
            currtime = time.time()
            query = query_pre_synapses_by_root_id_group(scanned_version_table, root_ids)
            result = query.collect()
            dt = time.time() - currtime
            rows.append(
                {
                    "method": method,
                    "version": version,
                    "query_id": i,
                    "n_result_rows": len(result),
                    "query_time_sec": dt,
                    "task": "pre_synapses_by_root_id_group",
                }
            )

# query induced synapses by root id group
n_trials = 10
n_roots_per_query = 100
select_root_id_sets = [
    base_table.select("post_pt_root_id")
    .unique()
    .sample(n_roots_per_query)
    .to_series()
    .to_list()
    for _ in range(n_trials)
]
for method, scanner in tqdm(
    method_scanners.items(),
    total=len(method_scanners),
    desc="induced_synapses_by_root_id_group",
):
    for version in versions:
        scanned_version_table = scanner(version)
        for i, root_ids in enumerate(select_root_id_sets):
            currtime = time.time()
            query = query_induced_synapses_by_root_id_group(
                scanned_version_table, root_ids
            )
            result = query.collect()
            dt = time.time() - currtime
            rows.append(
                {
                    "method": method,
                    "version": version,
                    "query_id": i,
                    "n_result_rows": len(result),
                    "query_time_sec": dt,
                    "task": "induced_synapses_by_root_id_group",
                }
            )

# query induced connections by root id group
n_trials = 10
n_roots_per_query = 100
select_root_id_sets = [
    base_table.select("post_pt_root_id")
    .unique()
    .sample(n_roots_per_query)
    .to_series()
    .to_list()
    for _ in range(n_trials)
]
for method, scanner in tqdm(
    method_scanners.items(),
    total=len(method_scanners),
    desc="induced_connections_by_root_id_group",
):
    for version in versions:
        scanned_version_table = scanner(version)
        for i, root_ids in enumerate(select_root_id_sets):
            currtime = time.time()
            query = query_induced_connections_by_root_id_group(
                scanned_version_table, root_ids
            )
            result = query.collect()
            dt = time.time() - currtime
            rows.append(
                {
                    "method": method,
                    "version": version,
                    "query_id": i,
                    "n_result_rows": len(result),
                    "query_time_sec": dt,
                    "task": "induced_connections_by_root_id_group",
                }
            )


result_table = pl.DataFrame(rows)


# %%

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("talk")

result_table = result_table.with_columns(pl.col("version").cast(pl.String))

for task in result_table["task"].unique():
    print(f"Task: {task}")
    task_table = result_table.filter(pl.col("task") == task)
    task_mean_result_table = (
        task_table.group_by(["method"])
        .agg(pl.col("query_time_sec").mean().alias("mean_query_time_sec"))
        .sort(["mean_query_time_sec"])
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.stripplot(
        data=task_table,
        x="method",
        order=task_mean_result_table["method"].to_list(),
        hue="version",
        y="query_time_sec",
        jitter=True,
        dodge=True,
        ax=ax,
    )
    sns.stripplot(
        data=task_mean_result_table,
        x="method",
        order=task_mean_result_table["method"].to_list(),
        y="mean_query_time_sec",
        color="black",
        size=60,
        marker="_",
        linewidth=3,
        ax=ax,
        legend=False,
    )
    ax.set_title(f"Task: {task}")
    ax.set(ylim=(0, None))

    # rotate x labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

# %%
result_table_by_method_task = result_table.group_by(["method", "task"]).agg(
    pl.col("query_time_sec").mean().alias("mean_query_time_sec"),
)
mean_result_table_by_method = (
    result_table_by_method_task.group_by("method")
    .agg(pl.col("mean_query_time_sec").mean().alias("overall_mean_query_time_sec"))
    .sort("overall_mean_query_time_sec")
)

fig, ax = plt.subplots(figsize=(12, 6))
sns.stripplot(
    data=result_table_by_method_task,
    x="method",
    hue="task",
    dodge=True,
    order=mean_result_table_by_method["method"].to_list(),
    y="mean_query_time_sec",
    jitter=True,
    ax=ax,
    legend=False,
)
sns.stripplot(
    data=mean_result_table_by_method,
    x="method",
    order=mean_result_table_by_method["method"].to_list(),
    y="overall_mean_query_time_sec",
    color="black",
    size=60,
    marker="_",
    linewidth=3,
    ax=ax,
    legend=False,
)
ax.set_title("Overall Mean Query Time by Method")
ax.set(ylim=(0, None))

# %%

fig, ax = plt.subplots(figsize=(12, 6))

method_order = (
    result_table.group_by("method")
    .agg(pl.col("query_time_sec").mean().alias("mean_query_time_sec"))
    .sort("mean_query_time_sec")["method"]
    .to_list()
)

task_order = (
    result_table.group_by("task")
    .agg(pl.col("query_time_sec").mean().alias("mean_query_time_sec"))
    .sort("mean_query_time_sec")["task"]
    .to_list()
)

sns.stripplot(
    data=result_table,
    x="method",
    hue="task",
    order=method_order,
    hue_order=task_order,
    dodge=True,
    y="query_time_sec",
    jitter=True,
    ax=ax,
)
sns.move_legend(
    ax,
    "upper left",
    bbox_to_anchor=(1.0, 1),
    title="Task",
    frameon=True,
)

# rotate x labels
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

fig, ax = plt.subplots(figsize=(12, 6))

sns.barplot(
    data=size_results,
    x="method",
    y="size_gb",
    order=method_order,
    ax=ax,
)
# rotate x labels
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
