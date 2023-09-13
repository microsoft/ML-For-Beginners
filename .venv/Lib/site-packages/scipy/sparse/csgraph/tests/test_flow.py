import numpy as np
from numpy.testing import assert_array_equal
import pytest

from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.csgraph import maximum_flow
from scipy.sparse.csgraph._flow import (
    _add_reverse_edges, _make_edge_pointers, _make_tails
)

methods = ['edmonds_karp', 'dinic']

def test_raises_on_dense_input():
    with pytest.raises(TypeError):
        graph = np.array([[0, 1], [0, 0]])
        maximum_flow(graph, 0, 1)
        maximum_flow(graph, 0, 1, method='edmonds_karp')


def test_raises_on_csc_input():
    with pytest.raises(TypeError):
        graph = csc_matrix([[0, 1], [0, 0]])
        maximum_flow(graph, 0, 1)
        maximum_flow(graph, 0, 1, method='edmonds_karp')


def test_raises_on_floating_point_input():
    with pytest.raises(ValueError):
        graph = csr_matrix([[0, 1.5], [0, 0]], dtype=np.float64)
        maximum_flow(graph, 0, 1)
        maximum_flow(graph, 0, 1, method='edmonds_karp')


def test_raises_on_non_square_input():
    with pytest.raises(ValueError):
        graph = csr_matrix([[0, 1, 2], [2, 1, 0]])
        maximum_flow(graph, 0, 1)


def test_raises_when_source_is_sink():
    with pytest.raises(ValueError):
        graph = csr_matrix([[0, 1], [0, 0]])
        maximum_flow(graph, 0, 0)
        maximum_flow(graph, 0, 0, method='edmonds_karp')


@pytest.mark.parametrize('method', methods)
@pytest.mark.parametrize('source', [-1, 2, 3])
def test_raises_when_source_is_out_of_bounds(source, method):
    with pytest.raises(ValueError):
        graph = csr_matrix([[0, 1], [0, 0]])
        maximum_flow(graph, source, 1, method=method)


@pytest.mark.parametrize('method', methods)
@pytest.mark.parametrize('sink', [-1, 2, 3])
def test_raises_when_sink_is_out_of_bounds(sink, method):
    with pytest.raises(ValueError):
        graph = csr_matrix([[0, 1], [0, 0]])
        maximum_flow(graph, 0, sink, method=method)


@pytest.mark.parametrize('method', methods)
def test_simple_graph(method):
    # This graph looks as follows:
    #     (0) --5--> (1)
    graph = csr_matrix([[0, 5], [0, 0]])
    res = maximum_flow(graph, 0, 1, method=method)
    assert res.flow_value == 5
    expected_flow = np.array([[0, 5], [-5, 0]])
    assert_array_equal(res.flow.toarray(), expected_flow)


@pytest.mark.parametrize('method', methods)
def test_bottle_neck_graph(method):
    # This graph cannot use the full capacity between 0 and 1:
    #     (0) --5--> (1) --3--> (2)
    graph = csr_matrix([[0, 5, 0], [0, 0, 3], [0, 0, 0]])
    res = maximum_flow(graph, 0, 2, method=method)
    assert res.flow_value == 3
    expected_flow = np.array([[0, 3, 0], [-3, 0, 3], [0, -3, 0]])
    assert_array_equal(res.flow.toarray(), expected_flow)


@pytest.mark.parametrize('method', methods)
def test_backwards_flow(method):
    # This example causes backwards flow between vertices 3 and 4,
    # and so this test ensures that we handle that accordingly. See
    #     https://stackoverflow.com/q/38843963/5085211
    # for more information.
    graph = csr_matrix([[0, 10, 0, 0, 10, 0, 0, 0],
                        [0, 0, 10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 10, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 10],
                        [0, 0, 0, 10, 0, 10, 0, 0],
                        [0, 0, 0, 0, 0, 0, 10, 0],
                        [0, 0, 0, 0, 0, 0, 0, 10],
                        [0, 0, 0, 0, 0, 0, 0, 0]])
    res = maximum_flow(graph, 0, 7, method=method)
    assert res.flow_value == 20
    expected_flow = np.array([[0, 10, 0, 0, 10, 0, 0, 0],
                              [-10, 0, 10, 0, 0, 0, 0, 0],
                              [0, -10, 0, 10, 0, 0, 0, 0],
                              [0, 0, -10, 0, 0, 0, 0, 10],
                              [-10, 0, 0, 0, 0, 10, 0, 0],
                              [0, 0, 0, 0, -10, 0, 10, 0],
                              [0, 0, 0, 0, 0, -10, 0, 10],
                              [0, 0, 0, -10, 0, 0, -10, 0]])
    assert_array_equal(res.flow.toarray(), expected_flow)


@pytest.mark.parametrize('method', methods)
def test_example_from_clrs_chapter_26_1(method):
    # See page 659 in CLRS second edition, but note that the maximum flow
    # we find is slightly different than the one in CLRS; we push a flow of
    # 12 to v_1 instead of v_2.
    graph = csr_matrix([[0, 16, 13, 0, 0, 0],
                        [0, 0, 10, 12, 0, 0],
                        [0, 4, 0, 0, 14, 0],
                        [0, 0, 9, 0, 0, 20],
                        [0, 0, 0, 7, 0, 4],
                        [0, 0, 0, 0, 0, 0]])
    res = maximum_flow(graph, 0, 5, method=method)
    assert res.flow_value == 23
    expected_flow = np.array([[0, 12, 11, 0, 0, 0],
                              [-12, 0, 0, 12, 0, 0],
                              [-11, 0, 0, 0, 11, 0],
                              [0, -12, 0, 0, -7, 19],
                              [0, 0, -11, 7, 0, 4],
                              [0, 0, 0, -19, -4, 0]])
    assert_array_equal(res.flow.toarray(), expected_flow)


@pytest.mark.parametrize('method', methods)
def test_disconnected_graph(method):
    # This tests the following disconnected graph:
    #     (0) --5--> (1)    (2) --3--> (3)
    graph = csr_matrix([[0, 5, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 9, 3],
                        [0, 0, 0, 0]])
    res = maximum_flow(graph, 0, 3, method=method)
    assert res.flow_value == 0
    expected_flow = np.zeros((4, 4), dtype=np.int32)
    assert_array_equal(res.flow.toarray(), expected_flow)


@pytest.mark.parametrize('method', methods)
def test_add_reverse_edges_large_graph(method):
    # Regression test for https://github.com/scipy/scipy/issues/14385
    n = 100_000
    indices = np.arange(1, n)
    indptr = np.array(list(range(n)) + [n - 1])
    data = np.ones(n - 1, dtype=np.int32)
    graph = csr_matrix((data, indices, indptr), shape=(n, n))
    res = maximum_flow(graph, 0, n - 1, method=method)
    assert res.flow_value == 1
    expected_flow = graph - graph.transpose()
    assert_array_equal(res.flow.data, expected_flow.data)
    assert_array_equal(res.flow.indices, expected_flow.indices)
    assert_array_equal(res.flow.indptr, expected_flow.indptr)


@pytest.mark.parametrize("a,b_data_expected", [
    ([[]], []),
    ([[0], [0]], []),
    ([[1, 0, 2], [0, 0, 0], [0, 3, 0]], [1, 2, 0, 0, 3]),
    ([[9, 8, 7], [4, 5, 6], [0, 0, 0]], [9, 8, 7, 4, 5, 6, 0, 0])])
def test_add_reverse_edges(a, b_data_expected):
    """Test that the reversal of the edges of the input graph works
    as expected.
    """
    a = csr_matrix(a, dtype=np.int32, shape=(len(a), len(a)))
    b = _add_reverse_edges(a)
    assert_array_equal(b.data, b_data_expected)


@pytest.mark.parametrize("a,expected", [
    ([[]], []),
    ([[0]], []),
    ([[1]], [0]),
    ([[0, 1], [10, 0]], [1, 0]),
    ([[1, 0, 2], [0, 0, 3], [4, 5, 0]], [0, 3, 4, 1, 2])
])
def test_make_edge_pointers(a, expected):
    a = csr_matrix(a, dtype=np.int32)
    rev_edge_ptr = _make_edge_pointers(a)
    assert_array_equal(rev_edge_ptr, expected)


@pytest.mark.parametrize("a,expected", [
    ([[]], []),
    ([[0]], []),
    ([[1]], [0]),
    ([[0, 1], [10, 0]], [0, 1]),
    ([[1, 0, 2], [0, 0, 3], [4, 5, 0]], [0, 0, 1, 2, 2])
])
def test_make_tails(a, expected):
    a = csr_matrix(a, dtype=np.int32)
    tails = _make_tails(a)
    assert_array_equal(tails, expected)
