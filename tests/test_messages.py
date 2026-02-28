from coflect.modules.hitl.common.messages import XaiRequestPayload


def test_xai_request_payload_defaults() -> None:
    payload = XaiRequestPayload(step=1, sample_idx=2, target_class=3, pred_class=4)
    as_dict = payload.to_dict()
    assert as_dict["backend"] == "torch"
    assert as_dict["request_kind"] == "periodic"
    assert as_dict["risk_score"] is None
    assert as_dict["horizon_epochs"] is None


def test_xai_request_payload_forecast_fields() -> None:
    payload = XaiRequestPayload(
        step=7,
        sample_idx=11,
        target_class=2,
        pred_class=4,
        request_kind="forecast",
        risk_score=0.73,
        horizon_epochs=10,
        backend="tensorflow",
    )
    as_dict = payload.to_dict()
    assert as_dict["request_kind"] == "forecast"
    assert as_dict["risk_score"] == 0.73
    assert as_dict["horizon_epochs"] == 10
    assert as_dict["backend"] == "tensorflow"
