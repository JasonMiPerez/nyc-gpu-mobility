"""Weighted taxi-zone PageRank and Louvain, with an explicit CPU reference.

The directed graph counts cross-zone trips. Louvain uses a separate undirected
projection whose weight is the sum of the two retained directional weights.
This is a small zone graph, not a claim that graph work needs a GPU.
"""

from __future__ import annotations

import hashlib
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .common import stable_id, write_json

EDGE_KEYS = ["pickup_zone_id", "dropoff_zone_id"]
DEFAULTS = {
    "min_trips_per_edge": 1,
    "pagerank_alpha": 0.85,
    "pagerank_tol": 1e-10,
    "pagerank_max_iter": 1000,
    "pagerank_atol": 1e-7,
    "pagerank_rtol": 1e-5,
    "louvain_resolution": 1.0,
    "louvain_threshold": 1e-7,
    "louvain_max_level": 100,
    "seed": 42,
}


def _options(config):
    options = {key: config.get("graph", {}).get(key, value) for key, value in DEFAULTS.items()}
    for key in ("min_trips_per_edge", "pagerank_max_iter", "louvain_max_level", "seed"):
        value = options[key]
        if isinstance(value, bool) or int(value) != value or value < (0 if key == "seed" else 1):
            raise ValueError(f"graph.{key} must be a valid nonnegative/positive integer")
        options[key] = int(value)
    for key in set(options) - {
        "min_trips_per_edge",
        "pagerank_max_iter",
        "louvain_max_level",
        "seed",
    }:
        options[key] = float(options[key])
        if not np.isfinite(options[key]) or options[key] <= 0:
            raise ValueError(f"graph.{key} must be finite and positive")
    if options["pagerank_alpha"] >= 1:
        raise ValueError("graph.pagerank_alpha must be less than one")
    return options


def _integer_column(values, name):
    result = pd.to_numeric(values, errors="raise")
    if result.isna().any() or not np.isfinite(result).all() or (result % 1 != 0).any():
        raise ValueError(f"{name} must contain finite integers")
    if not result.between(0, 2**53).all():
        raise ValueError(f"{name} must be nonnegative and exactly representable in float64")
    return result.astype("int64")


def _prepare(od, zones, options):
    missing = set([*EDGE_KEYS, "trip_count"]) - set(od)
    if missing or "zone_id" not in zones:
        raise ValueError(f"Graph inputs require zone_id and OD columns; missing {sorted(missing)}")
    nodes = zones.copy()
    # Match the pipeline's mapped-zone contract, while allowing coordinate-free fixtures.
    if "longitude" in nodes:
        nodes = nodes.loc[nodes.longitude.notna()].copy()
    nodes["zone_id"] = _integer_column(nodes.zone_id, "zone_id")
    if nodes.zone_id.duplicated().any():
        raise ValueError("Graph zone IDs must be unique")
    nodes = nodes.sort_values("zone_id").reset_index(drop=True)
    frame = od[[*EDGE_KEYS, "trip_count"]].copy()
    for name in frame:
        frame[name] = _integer_column(frame[name], name)
    total = sum(map(int, frame.trip_count))
    if total > 2**53:
        raise ValueError("Graph total weight exceeds exact float64 integer precision")
    known = frame.pickup_zone_id.isin(nodes.zone_id) & frame.dropoff_zone_id.isin(nodes.zone_id)
    pairs = frame.loc[known].groupby(EDGE_KEYS, as_index=False).trip_count.sum()
    pairs = pairs.loc[pairs.trip_count.gt(0)].copy()
    self_edges = pairs.loc[pairs.pickup_zone_id.eq(pairs.dropoff_zone_id)]
    cross = pairs.loc[pairs.pickup_zone_id.ne(pairs.dropoff_zone_id)]
    retained = cross.trip_count.ge(options["min_trips_per_edge"])
    edges = (
        cross.loc[retained]
        .sort_values(["trip_count", *EDGE_KEYS], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    undirected = (
        pd.DataFrame(
            {
                "zone_a": np.minimum(edges.pickup_zone_id, edges.dropoff_zone_id),
                "zone_b": np.maximum(edges.pickup_zone_id, edges.dropoff_zone_id),
                "trip_count": edges.trip_count,
            }
        )
        .groupby(["zone_a", "zone_b"], as_index=False)
        .trip_count.sum()
    )
    undirected = undirected.sort_values(
        ["trip_count", "zone_a", "zone_b"], ascending=[False, True, True]
    ).reset_index(drop=True)
    for endpoint, prefix in [("pickup_zone_id", "out"), ("dropoff_zone_id", "in")]:
        grouped = edges.groupby(endpoint).trip_count
        nodes[f"{prefix}_degree"] = nodes.zone_id.map(grouped.size()).fillna(0).astype("int64")
        nodes[f"{prefix}_trip_strength"] = (
            nodes.zone_id.map(grouped.sum()).fillna(0).astype("int64")
        )
    nodes["degree"] = nodes.in_degree + nodes.out_degree
    nodes["is_isolated"] = nodes.degree.eq(0)
    nodes["self_trip_count"] = (
        nodes.zone_id.map(self_edges.set_index("pickup_zone_id").trip_count)
        .fillna(0)
        .astype("int64")
    )
    counts = {
        "input_rows": len(od),
        "input_trip_count": total,
        "unmapped_trip_count": int(frame.loc[~known, "trip_count"].sum()),
        "mapped_zone_count": len(nodes),
        "unmapped_zone_rows": len(zones) - len(nodes),
        "self_trip_count": int(self_edges.trip_count.sum()),
        "self_zone_pair_count": len(self_edges),
        "cross_zone_pairs_before_threshold": len(cross),
        "removed_edge_count": int((~retained).sum()),
        "removed_trip_count": int(cross.loc[~retained, "trip_count"].sum()),
        "directed_edge_count": len(edges),
        "undirected_edge_count": len(undirected),
        "retained_trip_count": int(edges.trip_count.sum()),
        "isolated_zone_count": int(nodes.is_isolated.sum()),
    }
    assert counts["input_trip_count"] == sum(
        counts[key]
        for key in (
            "unmapped_trip_count",
            "self_trip_count",
            "removed_trip_count",
            "retained_trip_count",
        )
    )
    return nodes, edges, undirected, counts


def _networkx():
    try:
        import networkx as nx
    except ImportError as error:
        raise RuntimeError("Graph analysis requires NetworkX for the CPU reference") from error
    return nx


def _cpu_algorithms(vertices, edges, undirected, options):
    nx = _networkx()
    start = perf_counter()
    directed = nx.DiGraph()
    directed.add_nodes_from(vertices)
    directed.add_weighted_edges_from(
        edges[[*EDGE_KEYS, "trip_count"]].itertuples(index=False, name=None)
    )
    projection = nx.Graph()
    projection.add_nodes_from(vertices)
    projection.add_weighted_edges_from(undirected.itertuples(index=False, name=None))
    if not vertices:
        ranks, groups = {}, []
    elif edges.empty:
        ranks, groups = {v: 1 / len(vertices) for v in vertices}, [{v} for v in vertices]
    else:
        ranks = nx.pagerank(
            directed,
            alpha=options["pagerank_alpha"],
            weight="weight",
            max_iter=options["pagerank_max_iter"],
            tol=options["pagerank_tol"] / len(vertices),
            backend="networkx",
        )
        groups = nx.community.louvain_communities(
            projection,
            weight="weight",
            resolution=options["louvain_resolution"],
            threshold=options["louvain_threshold"],
            max_level=options["louvain_max_level"],
            seed=options["seed"],
            backend="networkx",
        )
    elapsed = perf_counter() - start
    modularity = (
        float(
            nx.community.modularity(
                projection,
                groups,
                weight="weight",
                resolution=options["louvain_resolution"],
                backend="networkx",
            )
        )
        if not undirected.empty
        else None
    )
    return ranks, groups, modularity, elapsed, projection


def _require_gpu():
    try:
        import cudf
        import cugraph
        import cupy as cp

        cp.cuda.runtime.deviceSynchronize()
    except Exception as error:
        raise RuntimeError(
            "GPU graph analysis requires working cuGraph/cuDF/CUDA; no CPU fallback"
        ) from error
    return cudf, cugraph, cp


def _gpu_algorithms(vertices, edges, undirected, options, modules):
    cudf, cugraph, cp = modules
    if edges.empty:
        # No graph algorithm is needed; explicitly recorded as a trivial result.
        return (
            {v: 1 / len(vertices) for v in vertices},
            [{v} for v in vertices],
            None,
            0.0,
            {"vertex_count": len(vertices), "directed_edge_count": 0, "undirected_edge_count": 0},
        )
    cp.cuda.runtime.deviceSynchronize()
    start = perf_counter()
    all_vertices = cudf.Series(vertices, dtype="int64")
    directed = cugraph.Graph(directed=True)
    edge_frame = edges.copy()
    edge_frame["trip_count"] = edge_frame.trip_count.astype("float64")
    directed.from_cudf_edgelist(
        cudf.from_pandas(edge_frame),
        source=EDGE_KEYS[0],
        destination=EDGE_KEYS[1],
        edge_attr="trip_count",
        renumber=True,
        store_transposed=True,
        vertices=all_vertices,
    )
    rank_frame = cugraph.pagerank(
        directed,
        alpha=options["pagerank_alpha"],
        tol=options["pagerank_tol"],
        max_iter=options["pagerank_max_iter"],
        fail_on_nonconvergence=True,
    ).to_pandas()
    projection = cugraph.Graph(directed=False)
    edge_frame = undirected.copy()
    edge_frame["trip_count"] = edge_frame.trip_count.astype("float64")
    projection.from_cudf_edgelist(
        cudf.from_pandas(edge_frame),
        source="zone_a",
        destination="zone_b",
        edge_attr="trip_count",
        renumber=True,
        vertices=all_vertices,
    )
    partition, modularity = cugraph.louvain(
        projection,
        max_level=options["louvain_max_level"],
        resolution=options["louvain_resolution"],
        threshold=options["louvain_threshold"],
    )
    partition = partition.to_pandas()
    cp.cuda.runtime.deviceSynchronize()
    elapsed = perf_counter() - start
    if rank_frame.vertex.duplicated().any() or partition.vertex.duplicated().any():
        raise ValueError("cuGraph returned duplicate graph vertices")
    ranks = dict(zip(map(int, rank_frame.vertex), map(float, rank_frame.pagerank)))
    groups = [set(map(int, group.vertex)) for _, group in partition.groupby("partition")]
    counts = {
        "vertex_count": int(directed.number_of_vertices()),
        "directed_edge_count": int(directed.number_of_edges()),
        "undirected_edge_count": int(projection.number_of_edges()),
    }
    return ranks, groups, float(modularity), elapsed, counts


def _communities(groups, vertices, isolates):
    seen = [int(v) for group in groups for v in group]
    if len(seen) != len(set(seen)) or set(seen) != set(vertices):
        raise ValueError("Community results must partition every mapped zone exactly once")
    # Isolates carry no modularity evidence; give each a separate display label.
    groups = [set(group) - isolates for group in groups]
    groups = [group for group in groups if group] + [{v} for v in sorted(isolates)]
    groups.sort(key=lambda group: min(group))
    return {vertex: label for label, group in enumerate(groups) for vertex in group}, groups


def analyze_graph(
    od_totals, zones, config, output_dir, backend="cpu", *, validate_cpu=True
) -> dict:
    """Analyze one supplied period/source selection; never reread individual trips."""
    if backend not in ("cpu", "gpu"):
        raise ValueError("Graph backend must be cpu or gpu")
    options = _options(config)
    nodes, edges, undirected, counts = _prepare(od_totals, zones, options)
    vertices = nodes.zone_id.tolist()
    modules = _require_gpu() if backend == "gpu" else None
    cpu_ranks, cpu_groups, cpu_modularity, cpu_elapsed, projection = (None, None, None, None, None)
    if backend == "cpu" or validate_cpu:
        cpu_ranks, cpu_groups, cpu_modularity, cpu_elapsed, projection = _cpu_algorithms(
            vertices, edges, undirected, options
        )
    ranks, groups, modularity, elapsed = cpu_ranks, cpu_groups, cpu_modularity, cpu_elapsed
    validation = {"status": "passed", "reference": "NetworkX CPU", "pagerank_max_abs_error": 0.0}
    if backend == "gpu" and not validate_cpu:
        validation = {
            "status": "awaiting_external_reference",
            "reference": "Separate CPU trial after timing",
            "pagerank_max_abs_error": None,
        }
    if backend == "gpu":
        ranks, groups, modularity, elapsed, device_counts = _gpu_algorithms(
            vertices, edges, undirected, options, modules
        )
        expected = {
            "vertex_count": len(nodes),
            **{key: counts[key] for key in ("directed_edge_count", "undirected_edge_count")},
        }
        if device_counts != expected:
            raise ValueError(
                f"GPU graph counts differ from the CPU reference: {device_counts} != {expected}"
            )
        validation["graph_counts"] = device_counts
    if set(ranks) != set(vertices):
        raise ValueError("PageRank results must contain every mapped zone exactly once")
    values = np.array([ranks[v] for v in vertices], dtype="float64")
    reference = (
        np.array([cpu_ranks[v] for v in vertices], dtype="float64")
        if cpu_ranks is not None
        else None
    )
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("PageRank returned invalid scores")
    if len(values) and not np.isclose(values.sum(), 1, atol=1e-6, rtol=0):
        raise ValueError("PageRank mass does not sum to one")
    if reference is not None and not np.allclose(
        values, reference, atol=options["pagerank_atol"], rtol=options["pagerank_rtol"]
    ):
        raise ValueError(
            "GPU PageRank differs from the NetworkX reference beyond declared tolerance"
        )
    if reference is not None:
        validation["pagerank_max_abs_error"] = (
            float(np.max(np.abs(values - reference))) if len(values) else 0.0
        )
    labels, groups = _communities(groups, vertices, set(nodes.loc[nodes.is_isolated, "zone_id"]))
    checked_modularity = (
        float(
            _networkx().community.modularity(
                projection,
                groups,
                weight="weight",
                resolution=options["louvain_resolution"],
                backend="networkx",
            )
        )
        if not undirected.empty and projection is not None
        else modularity
    )
    if modularity is not None and not np.isclose(
        modularity, checked_modularity, atol=1e-6, rtol=1e-5
    ):
        raise ValueError("Reported modularity does not match the weighted undirected projection")
    nodes["pagerank"] = values
    nodes["community"] = nodes.zone_id.map(labels).astype("int64")
    edge_hash = hashlib.sha256(edges.to_csv(index=False).encode()).hexdigest()
    graph_id = stable_id(
        "zonegraph", {"version": 1, "vertices": vertices, "edges_sha256": edge_hash, **options}
    )
    metadata = {
        "graph_id": graph_id,
        "status": "complete" if len(edges) else "no_cross_zone_edges",
        "backend": backend,
        "parameters": options,
        **counts,
        "graph_definition": "All mapped zones; directed origin→destination edges weighted by summed recorded trip counts for the supplied source/period.",
        "threshold_policy": "Sum repeated directed pairs first, remove self-zone trips, then retain directed pairs meeting min_trips_per_edge. The undirected projection sums the retained directions once per unordered pair.",
        "self_trip_policy": "Excluded from both algorithms and cross-zone degrees/strengths; retained separately as self_trip_count per zone and in totals.",
        "isolate_policy": "All mapped zones are included. PageRank uses uniform teleportation and dangling redistribution; zero-degree zones receive separate singleton community labels.",
        "pagerank_meaning": "Weighted directed connectivity score, normalized across mapped zones; not a passenger count or probability of an actual next trip.",
        "community_meaning": "Weighted undirected Louvain groups. IDs are arbitrary labels; CPU and GPU partitions may differ because optimization order differs.",
        "louvain_seed_policy": "The configured seed applies to NetworkX. The cuGraph Louvain API has no seed argument; identical GPU partitions across runs are not assumed.",
        "degree_meaning": "In/out degree counts distinct retained neighbors by direction; degree is their sum. In/out trip strength sums retained directional trip counts.",
        "edges_sha256": edge_hash,
        "pagerank_sum": float(values.sum()),
        "community_count": len(groups),
        "modularity": checked_modularity,
        "cpu_reference_modularity": cpu_modularity,
        "modularity_difference_from_cpu": checked_modularity - cpu_modularity
        if checked_modularity is not None and cpu_modularity is not None
        else None,
        "compute_elapsed_s": elapsed,
        "cpu_reference_elapsed_s": cpu_elapsed,
        "timing_scope": "One execution: graph construction, PageRank and Louvain, including GPU input/output transfers and completion. Excludes OD reduction, imports, validation, and artifact writes. Not a repeated benchmark or speedup claim.",
        "gpu_algorithms_executed": backend == "gpu" and not edges.empty,
        "preparation_backend": "CPU pandas for bounded OD reduction and degree/strength summaries",
        "software": {
            "networkx": version("networkx"),
            **({"cugraph": modules[1].__version__} if modules else {}),
        },
        "validation": validation,
        "limitations": "A graph of roughly 263 taxi zones is small; GPU acceleration is not assumed beneficial. Communities are descriptive connections, not neighborhoods, causal patterns, or individual travelers.",
        "api_sources": [
            "https://docs.rapids.ai/api/cugraph/stable/api_docs/api/cugraph/cugraph.Graph.from_cudf_edgelist/",
            "https://docs.rapids.ai/api/cugraph/stable/api_docs/api/cugraph/cugraph.pagerank/",
            "https://docs.rapids.ai/api/cugraph/stable/api_docs/api/cugraph/cugraph.louvain/",
        ],
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {}
    for name, table in [
        ("graph_nodes", nodes),
        ("graph_edges", edges),
        ("graph_undirected_edges", undirected),
    ]:
        tables[name] = f"{name}.parquet"
        table.to_parquet(output_dir / tables[name], index=False)
    write_json(output_dir / "graph_analysis.json", metadata)
    return {"tables": tables, "metadata": metadata}
