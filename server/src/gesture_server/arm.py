from __future__ import annotations

import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque

from gesture_shared import ActionRequest, ActionResponse, ActionState, ArmStatus, Gesture, Pose

log = logging.getLogger(__name__)

ACTION_TEXT = {
    Gesture.FIVE: "夹爪开合两次",
    Gesture.OK: "腕部左右旋转后回中",
    Gesture.SIX: "机械臂升降两次",
    Gesture.LIKE: "机械臂点头两次",
    Gesture.YES: "底座左右旋转后回中",
}


class ArmHardware(ABC):
    @abstractmethod
    def move_pose(self, pose: Pose) -> None: ...

    @abstractmethod
    def perform(self, gesture: Gesture) -> None: ...

    def close(self) -> None:
        pass


class DryRunArm(ArmHardware):
    def __init__(self, action_duration: float = 0.15):
        self.action_duration = action_duration
        self.calls: list[str] = []

    def move_pose(self, pose: Pose) -> None:
        self.calls.append(f"pose:{pose.value}")
        time.sleep(min(self.action_duration, 0.05))

    def perform(self, gesture: Gesture) -> None:
        self.calls.append(f"action:{gesture.value}")
        time.sleep(self.action_duration)


class DofbotArm(ArmHardware):
    SAFE_POSE = [90, 150, 20, 20, 90, 30]
    RECOGNITION_POSE = [90, 135, 22, 22, 90, 30]
    SERVO_TIME_MS = 500

    def __init__(self):
        import Arm_Lib  # type: ignore[import-not-found]
        self.arm = Arm_Lib.Arm_Device()

    def _write6(self, pose: list[int], duration: int = 500) -> None:
        self.arm.Arm_serial_servo_write6_array(pose, duration)
        time.sleep(duration / 1000 + 0.06)

    def _servo(self, servo: int, angle: int, duration: int = 500) -> None:
        self.arm.Arm_serial_servo_write(servo, angle, duration)
        time.sleep(duration / 1000 + 0.06)

    def move_pose(self, pose: Pose) -> None:
        self._write6(self.RECOGNITION_POSE if pose == Pose.RECOGNITION else self.SAFE_POSE, 1200)

    def perform(self, gesture: Gesture) -> None:
        self.move_pose(Pose.SAFE)
        if gesture == Gesture.FIVE:
            for _ in range(2):
                self._servo(6, 135); self._servo(6, 30)
        elif gesture == Gesture.OK:
            self._servo(5, 0); self._servo(5, 180); self._servo(5, 90)
        elif gesture == Gesture.SIX:
            for _ in range(2):
                self._write6([90, 130, 35, 25, 90, 30])
                self._write6([90, 155, 20, 20, 90, 30])
        elif gesture == Gesture.LIKE:
            for _ in range(2):
                self._servo(4, 55); self._servo(4, 20)
        elif gesture == Gesture.YES:
            self._servo(1, 60); self._servo(1, 120); self._servo(1, 90)
        self.move_pose(Pose.SAFE)


class ArmController:
    def __init__(self, hardware: ArmHardware, history_size: int = 512):
        self.hardware = hardware
        self._lock = threading.Lock()
        self._busy = False
        self._accepting = True
        self._current: str | None = None
        self._last: str | None = None
        self._error: str | None = None
        self._seen = deque(maxlen=history_size)
        self._seen_set: set[str] = set()
        self._thread: threading.Thread | None = None

    @property
    def status(self) -> ArmStatus:
        with self._lock:
            return ArmStatus(available=self._accepting, busy=self._busy,
                             current_action=self._current, last_action=self._last,
                             last_error=self._error)

    def _remember(self, request_id: str) -> None:
        if len(self._seen) == self._seen.maxlen:
            self._seen_set.discard(self._seen[0])
        self._seen.append(request_id)
        self._seen_set.add(request_id)

    def trigger(self, request: ActionRequest) -> ActionResponse:
        with self._lock:
            if request.request_id in self._seen_set:
                return ActionResponse(accepted=False, request_id=request.request_id,
                                      state=ActionState.REJECTED, reason="duplicate request_id")
            self._remember(request.request_id)
            if not self._accepting:
                return ActionResponse(accepted=False, request_id=request.request_id,
                                      state=ActionState.REJECTED, reason="arm unavailable")
            if self._busy:
                return ActionResponse(accepted=False, request_id=request.request_id,
                                      state=ActionState.REJECTED, reason="arm busy")
            action_id = str(uuid.uuid4())
            self._busy = True
            self._current = request.gesture.value
            self._error = None
            self._thread = threading.Thread(target=self._run, args=(request.gesture,),
                                            name=f"arm-{action_id[:8]}", daemon=True)
            self._thread.start()
            return ActionResponse(accepted=True, request_id=request.request_id,
                                  action_id=action_id, state=ActionState.ACCEPTED)

    def move_pose(self, pose: Pose) -> ActionResponse:
        request = ActionRequest(request_id=f"pose-{uuid.uuid4()}", gesture=Gesture.FIVE,
                                confidence=1.0)
        with self._lock:
            if not self._accepting or self._busy:
                return ActionResponse(accepted=False, request_id=request.request_id,
                                      state=ActionState.REJECTED, reason="arm busy or unavailable")
            action_id = str(uuid.uuid4())
            self._busy, self._current = True, f"pose:{pose.value}"
            self._thread = threading.Thread(target=self._run_pose, args=(pose,), daemon=True)
            self._thread.start()
            return ActionResponse(accepted=True, request_id=request.request_id,
                                  action_id=action_id, state=ActionState.ACCEPTED)

    def _run(self, gesture: Gesture) -> None:
        try:
            log.info("executing gesture=%s action=%s", gesture.value, ACTION_TEXT[gesture])
            self.hardware.perform(gesture)
            with self._lock:
                self._last = gesture.value
        except Exception as exc:
            log.exception("arm action failed")
            with self._lock:
                self._error = str(exc)
        finally:
            with self._lock:
                self._busy, self._current = False, None

    def _run_pose(self, pose: Pose) -> None:
        try:
            self.hardware.move_pose(pose)
            with self._lock:
                self._last = f"pose:{pose.value}"
        except Exception as exc:
            log.exception("pose action failed")
            with self._lock:
                self._error = str(exc)
        finally:
            with self._lock:
                self._busy, self._current = False, None

    def shutdown(self) -> None:
        with self._lock:
            self._accepting = False
        if self._thread:
            self._thread.join(timeout=10)
        try:
            self.hardware.move_pose(Pose.SAFE)
        except Exception:
            log.exception("failed to move arm to safe pose during shutdown")
        self.hardware.close()

