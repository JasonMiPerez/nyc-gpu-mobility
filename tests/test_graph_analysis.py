"""Graph contracts: real weights/direction, all zones, and strict GPU execution."""

import os

import numpy as np
import pandas as pd
import pytest

from mobility.models import graph_analysis as graph


def graph_inputs():
    zones = pd.DataFrame(
        {
            "zone_id": [10, 20, 30, 40, 50],
            "zone": ["A", "B", "C", "Isolated", "Unmapped"],
            "longitude": [-74.0, -73.9, -73.8, -73.7, None],
        }
    )
    od = pd.DataFrame(
        {
            "pickup_zone_id": [10, 10, 20, 20, 30, 10, 40, 50, 10],
            "dropoff_zone_id": [20, 20, 10, 30, 20, 10, 40, 10, 30],
            "trip_count": [3, 4, 2, 5, 1, 11, 9, 13, 0],
        }
    )
    return od, zones


def test_pair_reduction_threshold_self_trips_and_isolates(tmp_path):
    od, zones = graph_inputs()
    result = graph.analyze_graph(od, zones, {"graph": {"min_trips_per_edge": 3}}, tmp_path)
    nodes = pd.read_parquet(tmp_path / result["tables"]["graph_nodes"]).set_index("zone_id")
    edges = pd.read_parquet(tmp_path / result["tables"]["graph_edges"])
    assert set(nodes.index) == {10, 20, 30, 40}
    assert edges.trip_count.tolist() == [7, 5]  # Aggregate first, then threshold.
    assert nodes.loc[10, "out_trip_strength"] == 7
    assert nodes.loc[20, "in_degree"] == 1
    assert nodes.loc[20, "out_degree"] == 1
    assert nodes.loc[20, "degree"] == 2
    assert nodes.loc[40, "is_isolated"]
    assert nodes.loc[40, "self_trip_count"] == 9
    assert nodes.loc[40, "pagerank"] > 0
    assert nodes.loc[40, "community"] not in nodes.drop(40).community.tolist()
    m = result["metadata"]
    assert (m["input_trip_count"], m["unmapped_trip_count"], m["self_trip_count"]) == (48, 13, 20)
    assert (m["retained_trip_count"], m["removed_trip_count"], m["removed_edge_count"]) == (
        12,
        3,
        2,
    )
    assert m["isolated_zone_count"] == 1
    assert m["pagerank_sum"] == pytest.approx(1)


def test_undirected_projection_sums_both_directions_once(tmp_path):
    od, zones = graph_inputs()
    result = graph.analyze_graph(od, zones, {}, tmp_path)
    edges = pd.read_parquet(tmp_path / result["tables"]["graph_undirected_edges"])
    assert edges.to_dict("records") == [
        {"zone_a": 10, "zone_b": 20, "trip_count": 9},
        {"zone_a": 20, "zone_b": 30, "trip_count": 6},
    ]
    assert result["metadata"]["directed_edge_count"] == 4
    assert result["metadata"]["undirected_edge_count"] == 2


def test_pagerank_uses_weights_direction_and_ignores_self_trip_weight(tmp_path):
    import networkx as nx

    od, zones = graph_inputs()
    result = graph.analyze_graph(od, zones, {}, tmp_path / "first")
    nodes = pd.read_parquet(tmp_path / "first" / result["tables"]["graph_nodes"]).set_index(
        "zone_id"
    )
    reference = nx.DiGraph()
    reference.add_nodes_from([10, 20, 30, 40])
    reference.add_weighted_edges_from([(10, 20, 7), (20, 10, 2), (20, 30, 5), (30, 20, 1)])
    expected = nx.pagerank(reference, weight="weight", tol=1e-12, max_iter=1000, backend="networkx")
    for zone, value in expected.items():
        assert nodes.loc[zone, "pagerank"] == pytest.approx(value, abs=1e-9)
    assert nodes.loc[30, "pagerank"] > nodes.loc[10, "pagerank"]
    od.loc[od.pickup_zone_id.eq(od.dropoff_zone_id), "trip_count"] *= 1000
    changed = graph.analyze_graph(od, zones, {}, tmp_path / "changed")
    changed_nodes = pd.read_parquet(tmp_path / "changed" / changed["tables"]["graph_nodes"])
    np.testing.assert_allclose(changed_nodes.pagerank, nodes.pagerank)


@pytest.mark.parametrize("empty_zones", [False, True])
def test_empty_graph_has_explicit_trivial_results(tmp_path, empty_zones):
    od = pd.DataFrame(columns=["pickup_zone_id", "dropoff_zone_id", "trip_count"])
    zones = pd.DataFrame({"zone_id": [] if empty_zones else [1, 2]})
    result = graph.analyze_graph(od, zones, {}, tmp_path)
    nodes = pd.read_parquet(tmp_path / result["tables"]["graph_nodes"])
    assert result["metadata"]["status"] == "no_cross_zone_edges"
    assert result["metadata"]["modularity"] is None
    assert not result["metadata"]["gpu_algorithms_executed"]
    if not empty_zones:
        assert nodes.pagerank.tolist() == [0.5, 0.5]
        assert nodes.community.nunique() == 2
    else:
        assert nodes.empty


@pytest.mark.parametrize("invalid_count", [-1, np.nan, np.inf, 1.5])
def test_invalid_weights_fail_before_publication(tmp_path, invalid_count):
    od, zones = graph_inputs()
    od["trip_count"] = od.trip_count.astype(float)
    od.loc[0, "trip_count"] = invalid_count
    with pytest.raises(ValueError):
        graph.analyze_graph(od, zones, {}, tmp_path)
    assert not list(tmp_path.glob("*.parquet"))


def test_gpu_request_never_falls_back_to_cpu(monkeypatch, tmp_path):
    od, zones = graph_inputs()

    def unavailable():
        raise RuntimeError("cuGraph unavailable; no CPU fallback")

    monkeypatch.setattr(graph, "_require_gpu", unavailable)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        graph.analyze_graph(od, zones, {}, tmp_path, backend="gpu")
    assert not list(tmp_path.glob("*.parquet"))


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("MOBILITY_RUN_GPU_TESTS") != "1", reason="NVIDIA server required"
)
def test_cugraph_matches_cpu_graph_and_pagerank_including_isolate(tmp_path):
    od, zones = graph_inputs()
    result = graph.analyze_graph(od, zones, {}, tmp_path, backend="gpu")
    assert result["metadata"]["gpu_algorithms_executed"]
    assert result["metadata"]["validation"]["status"] == "passed"
    assert result["metadata"]["validation"]["pagerank_max_abs_error"] < 1e-7
    assert result["metadata"]["validation"]["graph_counts"] == {
        "vertex_count": 4,
        "directed_edge_count": 4,
        "undirected_edge_count": 2,
    }
