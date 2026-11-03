from shravan_ai.direction import (
    DIRECTION_BACKWARD,
    DIRECTION_FORWARD,
    DIRECTION_LATERAL,
    DIRECTION_NORMAL,
    DIRECTION_UNKNOWN,
    classify_direction,
)


def test_adl_task_code_is_always_normal_only():
    result = classify_direction("D01", "Forward walking on a flat surface")
    assert result.direction == DIRECTION_NORMAL
    assert result.source == "adl"


def test_forward_keyword_classified():
    result = classify_direction("F09", "Forward fall while walking caused by a trip")
    assert result.direction == DIRECTION_FORWARD
    assert result.source == "description"


def test_backward_keyword_classified():
    result = classify_direction("F05", "Backward fall while walking caused by a slip")
    assert result.direction == DIRECTION_BACKWARD


def test_lateral_keyword_classified():
    result = classify_direction("F08", "Lateral fall while walking caused by a slip")
    assert result.direction == DIRECTION_LATERAL


def test_unresolvable_description_flagged_not_guessed():
    result = classify_direction("F99", "Fall due to fainting")
    assert result.direction == DIRECTION_UNKNOWN
    assert result.source == "unresolved"
