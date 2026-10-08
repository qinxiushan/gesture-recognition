import pytest
from pydantic import ValidationError

from gesture_shared import ActionRequest, Gesture, PoseRequest


def test_action_contract_accepts_supported_gesture():
    request = ActionRequest(request_id="abc", gesture="thumbUp", confidence=0.9)
    assert request.gesture == Gesture.THUMB_UP


def test_action_contract_rejects_unknown_gesture():
    with pytest.raises(ValidationError):
        ActionRequest(request_id="abc", gesture="wave", confidence=0.9)


def test_contract_rejects_out_of_range_confidence_and_pose():
    with pytest.raises(ValidationError):
        ActionRequest(request_id="abc", gesture="ok", confidence=1.1)
    with pytest.raises(ValidationError):
        PoseRequest(pose="home")

