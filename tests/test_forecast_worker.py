from collections import deque

from coflect.modules.hitl.forecast.worker import SampleState, _overlap, _risk_score


def _state_from_rows(rows: list[tuple[float, float, float, bool, int]]) -> SampleState:
    state = SampleState(target_class=0, pred_class=0, last_step=0, history=deque(maxlen=16))
    for i, (epoch, p_true, margin, correct, pred_class) in enumerate(rows, start=1):
        state.push(
            epoch=epoch,
            p_true=p_true,
            margin=margin,
            correct=correct,
            pred_class=pred_class,
            step=i,
        )
    return state


def test_overlap_basic() -> None:
    assert _overlap([], [1, 2]) == 0.0
    assert _overlap([1, 2, 3], [2, 3, 4]) == 2 / 3


def test_risk_score_requires_minimum_history() -> None:
    state = _state_from_rows([(1.0, 0.6, 0.3, True, 1), (2.0, 0.62, 0.35, True, 1)])
    assert _risk_score(state) is None


def test_risk_score_higher_for_unstable_errorful_sample() -> None:
    good = _state_from_rows(
        [
            (1.0, 0.70, 0.40, True, 1),
            (2.0, 0.75, 0.45, True, 1),
            (3.0, 0.81, 0.50, True, 1),
            (4.0, 0.87, 0.55, True, 1),
        ]
    )
    bad = _state_from_rows(
        [
            (1.0, 0.45, 0.12, False, 2),
            (2.0, 0.31, 0.06, False, 3),
            (3.0, 0.24, 0.03, False, 1),
            (4.0, 0.20, 0.02, True, 2),
        ]
    )

    good_score = _risk_score(good)
    bad_score = _risk_score(bad)

    assert good_score is not None
    assert bad_score is not None
    assert bad_score > good_score
