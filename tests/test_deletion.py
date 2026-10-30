"""Check native exact betweenness and the macOS SDK compiler recovery."""
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import networkx as nx
import numpy as np
import pytest

from scripts.recompute_deletion_700 import compile_brandes, exact_betweenness, read_original_evidence

CPP = Path(__file__).resolve().parents[1] / 'scripts/brandes_exact.cpp'


@pytest.mark.parametrize('workers', [1, 4])
def test_exact_betweenness_matches_networkx(workers):
    graph = nx.cycle_graph(6)
    graph.add_edge(0, 6)
    graph.add_node(7)  # Disconnected vertex and multiple shortest paths.
    for current in (graph, nx.subgraph_view(graph, filter_node=lambda v: v != 0)):
        nodes = sorted(current)
        actual = exact_betweenness(current, nodes, CPP, workers)
        expected = nx.betweenness_centrality(current, normalized=True, weight=None)
        np.testing.assert_allclose(
            [actual[v] for v in nodes], [expected[v] for v in nodes],
            rtol=0, atol=1e-12,
        )


def test_missing_apple_headers_retry(tmp_path):
    headers = tmp_path / 'usr/include/c++/v1'
    headers.mkdir(parents=True)
    (headers / 'algorithm').touch()
    responses = [
        CompletedProcess([], 1, '', "fatal error: 'algorithm' file not found"),
        CompletedProcess([], 0, str(tmp_path) + '\n', ''),
        CompletedProcess([], 0, '', ''),
    ]
    with patch('scripts.recompute_deletion_700.sys.platform', 'darwin'), \
         patch('scripts.recompute_deletion_700.shutil.which', side_effect=lambda x: '/usr/bin/' + x), \
         patch('scripts.recompute_deletion_700.subprocess.run', side_effect=responses) as run:
        compile_brandes(CPP, tmp_path / 'exe')
    assert run.call_count == 3
    assert run.call_args.args[0][-2:] == ['-isystem', str(headers)]


def test_unrelated_compilation_failure_is_not_hidden(tmp_path):
    with patch('scripts.recompute_deletion_700.shutil.which', return_value='/usr/bin/c++'), \
         patch('scripts.recompute_deletion_700.subprocess.run',
               return_value=CompletedProcess([], 1, '', 'syntax error')) as run:
        with pytest.raises(RuntimeError, match='syntax error'):
            compile_brandes(CPP, tmp_path / 'exe')
    assert run.call_count == 1


def test_empty_integrated_evidence_has_actionable_error(tmp_path):
    evidence = tmp_path / 'empty.csv'
    evidence.touch()
    with pytest.raises(ValueError, match='empty or missing string_id'):
        read_original_evidence(evidence)
