import numpy as np

from scripts.test_grape_retention import summarize_trace


def test_retention_summary_distinguishes_held_and_dropped_grapes():
    phase_1d = np.array([0.40, 0.50, 0.70, 0.81, 0.90, 0.99])
    phase = np.repeat(phase_1d[:, None], 2, axis=1)
    # Trial 0 reaches and holds 11 cm. Trial 1 briefly rises, then drops.
    height = np.array(
        [
            [0.01, 0.01],
            [0.03, 0.03],
            [0.08, 0.07],
            [0.11, 0.06],
            [0.11, 0.02],
            [0.11, 0.01],
        ]
    )
    distance = np.array(
        [
            [0.01, 0.01],
            [0.01, 0.01],
            [0.01, 0.02],
            [0.01, 0.05],
            [0.01, 0.10],
            [0.01, 0.12],
        ]
    )

    out = summarize_trace(
        phase,
        height,
        distance,
        final_height_threshold=0.10,
        distance_threshold=0.03,
    )

    assert out["success"].tolist() == [True, False]
    assert out["success_rate"] == 0.5
    assert np.allclose(out["max_height"], [0.11, 0.07])
    assert np.allclose(out["final_height"], [0.11, 0.01])
