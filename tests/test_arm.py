import time

from gesture_shared import ActionRequest, ActionState
from gesture_server.arm import ArmController, DryRunArm


def request(request_id: str, gesture: str = "like") -> ActionRequest:
    return ActionRequest(request_id=request_id, gesture=gesture, confidence=0.9)


def test_arm_rejects_busy_and_duplicate_requests():
    controller = ArmController(DryRunArm(action_duration=0.1))
    first = controller.trigger(request("one"))
    busy = controller.trigger(request("two"))
    duplicate = controller.trigger(request("one"))
    assert first.accepted
    assert not busy.accepted and busy.reason == "arm busy"
    assert not duplicate.accepted and duplicate.reason == "duplicate request_id"
    time.sleep(0.15)
    assert controller.status.last_action == "like"


def test_arm_accepts_after_previous_action_finishes():
    controller = ArmController(DryRunArm(action_duration=0.01))
    assert controller.trigger(request("one")).state == ActionState.ACCEPTED
    time.sleep(0.03)
    assert controller.trigger(request("two", "yes")).accepted

