from signbridge.common.landmarks import (
    FACE,
    HAND_CONNECTIONS,
    LEFT_HAND,
    LIPS,
    POSE,
    POSE_UPPER_BODY,
    RIGHT_HAND,
    ROWS_PER_FRAME,
)


def test_slices_are_contiguous_and_cover_all() -> None:
    assert FACE[0] == 0
    assert FACE[1] == LEFT_HAND[0]
    assert LEFT_HAND[1] == POSE[0]
    assert POSE[1] == RIGHT_HAND[0]
    assert RIGHT_HAND[1] == ROWS_PER_FRAME
    
    total_len = (FACE[1] - FACE[0]) + (LEFT_HAND[1] - LEFT_HAND[0]) + \
                (POSE[1] - POSE[0]) + (RIGHT_HAND[1] - RIGHT_HAND[0])
    
    assert total_len == ROWS_PER_FRAME
    assert ROWS_PER_FRAME == 543

def test_hand_connections_indices() -> None:
    for i, j in HAND_CONNECTIONS:
        assert 0 <= i <= 20
        assert 0 <= j <= 20

def test_lips_unique_indices() -> None:
    assert len(set(LIPS)) == 40
    for idx in LIPS:
        assert 0 <= idx < 468

def test_pose_upper_body_indices() -> None:
    for idx in POSE_UPPER_BODY:
        assert 0 <= idx <= 32
