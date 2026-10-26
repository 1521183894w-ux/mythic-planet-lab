"""多洞重叠供体预算、顺序无关和既有非短缺结果。"""
import numpy as np
import pytest

from mythic.tectonics.borrowing import cap_shared_borrow


def test_overlapping_holes_share_one_capacity_without_order_dependence():
    nb = np.array([[0, 1], [0, 2], [1, 2]])
    requests = np.array([[9., 2.], [6., 8.], [6., 4.]])
    capacity = np.array([10., 5., 20.])
    share = cap_shared_borrow(nb, requests, capacity)
    used = np.bincount(nb.ravel(), share.ravel(), minlength=3)
    assert np.all(used <= capacity + 1e-12)
    assert np.all(share.sum(axis=1) <= requests.sum(axis=1))
    assert share[0, 0] / share[1, 0] == pytest.approx(9. / 6.)
    order = [2, 0, 1]
    np.testing.assert_allclose(cap_shared_borrow(nb[order], requests[order], capacity), share[order], atol=1e-12)
    assert (requests - share).sum() == pytest.approx(requests.sum() - used.sum())


def test_no_shortage_and_empty_requests_preserve_original_allocations_exactly():
    nb = np.array([[0, 1], [0, 2]])
    requests = np.array([[2., 3.], [4., 5.]])
    np.testing.assert_array_equal(cap_shared_borrow(nb, requests, np.array([10., 10., 10.])), requests)
    assert cap_shared_borrow(np.empty((0, 2), int), np.empty((0, 2)), np.ones(3)).shape == (0, 2)


def test_duplicate_donor_within_one_hole_and_zero_capacity_are_capped_together():
    nb = np.array([[0, 0, 1]])
    share = cap_shared_borrow(nb, np.array([[5., 5., 3.]]), np.array([6., 0.]))
    np.testing.assert_allclose(share, [[3., 3., 0.]])
    with pytest.raises(ValueError):
        cap_shared_borrow(nb, np.array([[5., np.nan, 3.]]), np.array([6., 0.]))
