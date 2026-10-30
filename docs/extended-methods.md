# Duration, graph, and full-pipeline methods

These methods describe the implemented calculations. Performance numbers belong to completed, validated experiment artifacts; this document does not infer results from trials still running.

## Duration alerts

The observed value is the mean duration, in minutes, of eligible trips starting in a particular taxi zone and hour. Durations are summed across batches and divided by the total trip count, so small batches do not receive extra weight. An hour needs at least 20 eligible trips to contribute to a baseline or receive a score. Empty hours are absent, and low-count hours remain explicitly unsupported; neither becomes a zero-minute observation.

Trips use the recorded quality flags and the fixed study bounds: at most 100 miles, 180 minutes, and 100 mph average speed. Zero reported distance remains possible. These restrictions affect duration and clustering analyses; valid pickup/drop-off counts retain their separate eligibility rules. Ambiguous or nonexistent daylight-saving hours are excluded.

January–March supplies the baseline. For each zone, weekday, and hour, the detector takes the median of the sufficiently populated hourly means. Each historical hour has equal weight, and at least eight matching hours are required. Its score is:

```text
score = (observed mean minutes − historical median minutes)
        / max(1.4826 × historical MAD, 2 minutes)
```

An alert also requires an absolute change of at least five minutes. April selects the smallest candidate threshold at least the configured default of four whose calibration alert fraction meets the 1% budget. If none qualifies, the largest candidate is retained and the unmet budget is reported. With no usable April history, the default remains and calibration is explicitly unavailable. May–June is scored with the baseline and selected threshold frozen.

Adjacent alert hours with the same zone, direction, and split form a candidate incident. Unsupported hours break an incident. Summing deviations in hourly means does **not** produce total passenger delay. Different routes, trip distances, or destination mixes can change average duration, so these alerts alone cannot establish congestion or its cause. The fixed study restrictions were introduced after inspecting historical data; further independent data is needed for prospective confirmation.

## Taxi-zone graph

All mapped taxi zones are vertices, including zones with no retained connections. Repeated origin–destination pairs are summed. Same-zone trips are counted separately and removed from the graph algorithms. The minimum-trip threshold applies to each directed pair after summation; the default of one retains every positive cross-zone pair.

Weighted PageRank uses the directed graph, with damping 0.85, uniform teleportation, and uniform redistribution from dangling vertices. Its scores sum to one across mapped zones. They describe weighted connectivity in this constructed network; they are not passenger counts or an observed next-trip probability.

Louvain uses an undirected projection: A→B and B→A weights are added once for the unordered pair. It seeks communities with high weighted modularity. Community numbers are arbitrary. CPU and GPU optimizers can produce different valid partitions; the cuGraph Louvain API does not provide the CPU implementation's seed control. Isolated zones receive separate singleton display labels.

GPU execution is explicit cuGraph, with no CPU fallback. Validation checks the retained graph counts, every vertex, PageRank mass and numerical agreement with NetworkX. Modularity is independently recomputed for the returned partition. The GPU production path still runs the NetworkX reference as part of its correctness checks. This small graph does not automatically benefit from GPU execution.

## Full-pipeline experiments

The default experiment uses January, January–March, and January–June source prefixes. Each backend receives the same source checksums, configuration, sample identities, and preprocessing policy. Every trial runs in a fresh process, with four CPU threads by default. Each size has one discarded full warmup per backend and three measured repetitions; backend order is randomized within each repetition.

The stopwatch surrounds the production pipeline: source checksums and reading, CPU normalization, feature work, aggregation, model fitting and evaluation, prediction, duration and graph analyses, artifact writes, and built-in reconciliation. Downloads, publication, worker bootstrap, GPU dependency preflight, and the later cross-trial output comparison are outside it. Lazy imports and initialization reached inside the production function remain timed. Filesystem and CUDA compilation caches are not reset.

The expensive per-batch CPU aggregation reference is disabled in these benchmark trials. After timing, aggregate keys and integer counts must agree exactly; floating sums use `rtol=1e-9` and `atol=1e-7`. Source identities, model eligibility, deterministic sample hashes, and scaler inputs are checked. Fitted cluster labels are not required to match; selected group counts and quality diagnostics are retained. The graph's internal CPU reference remains inside the timer. Missing or failed backend runs cannot become a reported speedup.

January and January–March have no April calibration or May–June test data. Their model metadata reports this explicitly. Growth across these prefixes therefore includes additional evaluation work as well as additional records.

Host RSS is sampled for the worker and descendants every requested 10 ms; NVML process memory and device utilization every requested 20 ms. Actual sample counts, maximum gaps, unavailable readings, and process-tree coverage are saved. These are sampled peaks, which can miss short spikes. RSS can double-count shared pages. GPU process memory includes context and allocator reservations, and device utilization is shared across processes. Other GPU processes are recorded as possible contention; desktop processes are not stopped.

Monitoring also consumes CPU time on the shared host. The reported runtimes describe this monitored setup; no adjustment for observer overhead is applied.

## Interpreting total runtime

A faster GPU stage can coexist with similar overall CPU/GPU times. Both pipelines perform CPU timestamp validation, canonical Parquet writes and rereads, deterministic sampling, feature/scaler transformations, anomaly calculations, and output serialization. Clustering scans canonical data for samples and again for prediction; duration analysis makes another scan. GPU ingestion features are not reused as the clustering input: clustering computes its own CPU features before cuML fitting/prediction. Transfers and GPU initialization add work as well.

For a completed fresh full-pipeline trial, `pipeline_timings` measures each source's ingestion/normalization/write/aggregation block. Clustering metadata records candidate fitting and prediction scopes. Graph metadata separates its algorithm execution from its CPU-reference diagnostic time. These fields help locate costs, but do not cover every operation or isolate CPU time. Any unassigned remainder also includes model selection, stability fits, merging, validation, and writes. A model-only refit's inherited ingestion timings belong to its original run and cannot be subtracted from the refit time. Use the repeated whole-pipeline measurements for the overall comparison, and compare earlier stage benchmarks only within their own recorded workload and scope.
