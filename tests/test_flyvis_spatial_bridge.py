"""CPU-only contracts for the online FlyVis spatial bridge helpers."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _bridge():
    path = (
        Path(__file__).parents[1]
        / "integrations/super_mario/flyvis_spatial_bridge.py"
    )
    spec = importlib.util.spec_from_file_location("flyvis_spatial_bridge_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_canonical_pca_is_deterministic_and_two_dimensional():
    bridge = _bridge()
    points = np.asarray([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.1], [2.0, 1.0, 0.2],
        [3.0, 1.0, 0.3], [4.0, 2.0, 0.4],
    ])
    center, basis = bridge.canonical_pca(points)
    center_again, basis_again = bridge.canonical_pca(points)
    assert center.shape == (3,)
    assert basis.shape == (2, 3)
    np.testing.assert_allclose(center, center_again)
    np.testing.assert_allclose(basis, basis_again)
    np.testing.assert_allclose(basis @ basis.T, np.eye(2), atol=1e-6)


def test_quantile_bins_saturate_outliers():
    bridge = _bridge()
    bins = bridge.quantile_bins(np.asarray((-100.0, 0.0, 1.0, 2.0, 100.0)), 4)
    assert bins.tolist()[0] == 0
    assert bins.tolist()[-1] == 3
    assert np.all(np.diff(bins) >= 0)


@pytest.mark.parametrize("cell_type", list(("R1", "L1", "L2", "T4a", "T5d")))
def test_normalized_drive_is_bounded_and_monotonic(cell_type):
    bridge = _bridge()
    values = np.linspace(-3, 3, 21, dtype=np.float32)
    drive = bridge.normalized_drive(values, cell_type, maximum=0.18)
    assert drive.shape == values.shape
    assert np.all(drive >= 0)
    assert np.all(drive <= 0.18)
    assert np.all(np.diff(drive) >= 0)


def test_bridge_preserves_static_and_directional_cell_types():
    bridge = _bridge()
    assert set(bridge.STATIC_TYPES) == {"R1", "L1", "L2"}
    assert set(bridge.MOTION_TYPES) == {
        "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"
    }
    assert bridge.FLYVIS_TO_MALE_CNS["R1"] == "R1-6"


def test_policy_readout_retains_positions_and_ablates_both_feature_groups():
    bridge = _bridge()
    grid = np.arange(528, dtype=np.float32).reshape(11, 6, 8) / 528 - 0.5
    normal = bridge.policy_spatial_features(grid).reshape(11, 6, 8)
    np.testing.assert_allclose(normal, np.tanh(grid))
    assert np.count_nonzero(bridge.policy_spatial_features(grid, "all_visual")) == 0
    static_off = bridge.policy_spatial_features(grid, "flyvis_static").reshape(11, 6, 8)
    motion_off = bridge.policy_spatial_features(grid, "flyvis_motion").reshape(11, 6, 8)
    assert np.count_nonzero(static_off[:3]) == 0
    np.testing.assert_array_equal(static_off[3:], normal[3:])
    assert np.count_nonzero(motion_off[3:]) == 0
    np.testing.assert_array_equal(motion_off[:3], normal[:3])


def test_spatial_injections_preserve_bins_and_support_stage_ablations():
    bridge = _bridge()
    grids = np.zeros((len(bridge.FLYVIS_TO_MALE_CNS), 2, 4), dtype=np.float32)
    targets = {}
    identifier = 0
    for cell_type in bridge.FLYVIS_TO_MALE_CNS:
        for side in ("L", "R"):
            for row in range(2):
                for column in range(2):
                    targets[(cell_type, side, row, column)] = np.asarray([identifier])
                    identifier += 1

    normal, stats = bridge.spatial_injections(grids, targets)
    without_static, _ = bridge.spatial_injections(
        grids, targets, ablation="flyvis_static"
    )
    without_motion, _ = bridge.spatial_injections(
        grids, targets, ablation="flyvis_motion"
    )

    assert len(normal) == 11 * 2 * 2 * 2
    assert len(without_static) == 8 * 2 * 2 * 2
    assert len(without_motion) == 3 * 2 * 2 * 2
    assert stats["flyvis_spatial_injection_groups"] == len(normal)
