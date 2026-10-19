from deployment.l3_stage_hard import _split_long_segments


def test_tactical_refinement_evenly_splits_long_boundary_units() -> None:
    refined = _split_long_segments(
        [(0, 9, "stable"), (10, 12, "transition")],
        max_frames=4,
    )
    assert refined == [
        (0, 2, "stable"),
        (3, 5, "stable"),
        (6, 9, "stable"),
        (10, 12, "transition"),
    ]
