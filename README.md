# parqlayground

Investigating different strategies for storing large tables (e.g. the synapse table) in
parquet/deltalakes.

## Strategies

- **Naive-copy:** Store a single delta table per materialization version, making a new one each time.
- **Split-seg:** Store a delta table of synapse ID, spatial info. Store a separate one for segmentation
  info with synapse ID also. Join as needed.
- **Split-seg-multi:** Store a delta table of synapse ID, spatial info. Store two separate tables, one each for both pre and
  postsynaptic segmentation (each with synapse IDs). Join as needed.
- **Delta-columns:** Store a single delta table, append columns per new materialization version as needed.
- **Delta-rows:** Store a single delta table, but as segmentation changes, update rows. Use delta lake
  versioning patterns to access prior versions.

## Parameters

- Partition key (post root ID, pre root ID, synapse ID, etc)
- Number of partitions (0 partitions ~= a parquet file)
- Bloom filters
- File size

## Metrics

### Storage size

- Total size of the combined tables on cloud storage

### Query time

Example queries to test and time (querying from the cloud):

- Get a single postsynapse by ID
- Get a single presynapse by ID
- Get all postsynapses for a root ID
- Get all presynapses for a root ID
- Get all postsynapses for ~100 root IDs
- Get all presynapses for ~100 root IDs
- Get all synapses among ~100 root IDs
- Get the mean postsynapse size for a root ID
- Get the connections among cells with nuclei
