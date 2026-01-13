# %%
import time
from typing import Union
from caveclient import CAVEclient
import polars as pl
from cloudpathlib import AnyPath as Path


def scan_table(table_path: Union[Path, str]) -> pl.LazyFrame:
    table_path = Path(table_path)
    if table_path.suffix == ".parquet":  # ty: ignore
        return pl.scan_parquet(str(table_path))
    else:
        return pl.scan_delta(str(table_path))


base_path = Path(
    "/Users/ben.pedigo/code/meshrep/meshrep/data/synapses_pni_2_v1412_deltalake"
)

table = scan_table(base_path)

table.collect_schema()

# %%
# setup needed for each query type
#
# get_single_synapse_by_id /
#   list of k synapse ids to query for
#
# get_post_synapses_by_root_id /
# get_pre_synapses_by_root_id /
# get_post_synapses_mean_size_by_root_id /
# get_pre_synapses_mean_size_by_root_id:
#   list of k root ids to query for
#
# get_post_synapses_by_root_id_group /
# get_pre_synapses_by_root_id_group /
# get_induced_synapses_by_root_id_group /
# get_induced_connections_by_root_id_group:
#   list of m groups of root ids to query for, each group containing k root ids, say 100
#


def query_single_synapse_by_id(table: pl.LazyFrame, synapse_id: int) -> pl.LazyFrame:
    return table.filter(pl.col("synapse_id") == synapse_id)


def query_many_synapses_by_ids(
    table: pl.LazyFrame, synapse_ids: list[int]
) -> pl.LazyFrame:
    return table.filter(pl.col("synapse_id").is_in(synapse_ids))


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

seed = 8888
select_synapse_ids = table.select("synapse_id").collect().sample(10000, seed=seed)

for syn_id in select_synapse_ids.to_series()[:20]:
    query = query_single_synapse_by_id(table, syn_id)
    currtime = time.time()
    result = query.collect()
    print(f"{time.time() - currtime:.3f} seconds elapsed.")

# %%

select_synapse_ids.write_parquet("data/select_synapse_ids.parquet")

# %%



version = 1412
TABLE_CACHE_PATH = Path("/Users/ben.pedigo/code/meshrep/meshrep/data/table_cache")
table_path = TABLE_CACHE_PATH / f"v{version}" / "aibs_cell_info.csv.gz"  # ty: ignore

table: pl.DataFrame = pl.read_csv(table_path)
neuron_table = table.filter(
    pl.col("cell_type_source") == "allen_v1_column_types_slanted_ref"
)
sv_ids = neuron_table.select("pt_supervoxel_id").to_series().to_list()

versions = [343, 943, 1412, 1507]

client = CAVEclient("minnie65_phase3_v1")
for version in versions:
    timestamp = client.materialize.get_timestamp(version)  # ty: ignore
    root_ids_at_version = client.chunkedgraph.get_roots(  # ty: ignore
        sv_ids, timestamp=timestamp
    )
    neuron_table = neuron_table.with_columns(
        pl.Series(
            f"root_id_at_{version}",
            root_ids_at_version,
        )
    )

neuron_table.write_parquet("data/column_root_info.parquet")

# %%


# %%
table_path = "gs://allen-minnie-phase3/bdp-deltalakes/minnie65_phase3_v1/v343/synapses_pni_2_v1_filtered_view"
table = scan_table(table_path)
