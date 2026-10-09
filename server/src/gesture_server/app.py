import asyncio
import logging
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from gesture_shared import (
    API_VERSION, ActionRequest, ActionResponse, CameraStatus, HealthResponse,
    Pose, PoseRequest, StatusResponse,
)

from .arm import ArmController, DofbotArm, DryRunArm
from .camera import CameraSource, MockCamera, OpenCVCamera
from .config import ServerSettings

log = logging.getLogger(__name__)


class Runtime:
    def __init__(self, camera: CameraSource, arm: ArmController):
        self.camera = camera
        self.arm = arm


def create_app(settings: Optional[ServerSettings] = None,
               runtime: Optional[Runtime] = None) -> FastAPI:
    settings = settings or ServerSettings()
    if runtime is None:
        camera: CameraSource = (MockCamera(settings.width, settings.height, settings.jpeg_quality)
                                if settings.mock_camera else
                                OpenCVCamera(settings.camera_source, settings.width, settings.height,
                                             settings.fps, settings.jpeg_quality))
        hardware = DryRunArm() if settings.dry_run else DofbotArm()
        runtime = Runtime(camera=camera, arm=ArmController(hardware))

    app = FastAPI(title="Gesture Robot Server", version="0.1.0")
    app.state.runtime = runtime

    @app.on_event("startup")
    def startup():
        runtime.camera.start()
        runtime.arm.move_pose(Pose(settings.start_pose))

    @app.on_event("shutdown")
    def shutdown():
        runtime.camera.stop()
        runtime.arm.shutdown()

    @app.get(f"/api/{API_VERSION}/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        rt: Runtime = request.app.state.runtime
        camera = CameraStatus(available=rt.camera.available, device=rt.camera.device,
                              last_error=rt.camera.last_error)
        return HealthResponse(status="ok" if camera.available and rt.arm.status.available else "degraded",
                              camera=camera, arm=rt.arm.status)

    @app.get(f"/api/{API_VERSION}/status", response_model=StatusResponse)
    def status(request: Request) -> StatusResponse:
        return StatusResponse(arm=request.app.state.runtime.arm.status)

    async def frames(camera: CameraSource):
        while True:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, camera.get_jpeg, 2.0)
            if data:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                       str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
            else:
                await asyncio.sleep(0.1)

    @app.get(f"/api/{API_VERSION}/video/stream")
    def video(request: Request) -> StreamingResponse:
        return StreamingResponse(frames(request.app.state.runtime.camera),
                                 media_type="multipart/x-mixed-replace; boundary=frame")

    @app.post(f"/api/{API_VERSION}/actions", response_model=ActionResponse)
    def action(payload: ActionRequest, request: Request) -> ActionResponse:
        log.info("action request id=%s gesture=%s confidence=%.3f", payload.request_id,
                 payload.gesture.value, payload.confidence)
        return request.app.state.runtime.arm.trigger(payload)

    @app.post(f"/api/{API_VERSION}/arm/pose", response_model=ActionResponse)
    def pose(payload: PoseRequest, request: Request) -> ActionResponse:
        return request.app.state.runtime.arm.move_pose(payload.pose)

    return app
