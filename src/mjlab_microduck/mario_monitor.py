"""Live Mario framebuffer transport and Microduck head-camera rendering.

The emulator and BAM/Mjlab environments require different Python versions, so
RGB frames cross the process boundary through a small shared-memory seqlock.
The reader always copies a complete frame; it never exposes memory that the
sidecar may be changing underneath the renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import shared_memory
import struct
from typing import Final

import mujoco
import numpy as np


DEFAULT_FRAME_SHM: Final = "microduck_mario_rgb"
FRAME_MAGIC: Final = b"MDMRGB1\0"
FRAME_HEADER: Final = struct.Struct("<8sIIIIQ")
MARIO_TEXTURE_NAME: Final = "mario_screen_texture"
HEAD_CAMERA_NAME: Final = "head_camera"


@dataclass(frozen=True, slots=True)
class MarioFrame:
    """One consistent RGB frame copied from the emulator process."""

    sequence: int
    rgb: np.ndarray


class MarioFrameSubscriber:
    """Read the newest sidecar frame from named shared memory."""

    def __init__(self, name: str = DEFAULT_FRAME_SHM) -> None:
        self.name = name
        self._shm: shared_memory.SharedMemory | None = None

    @property
    def connected(self) -> bool:
        return self._shm is not None

    def connect(self) -> None:
        if self._shm is None:
            self._shm = shared_memory.SharedMemory(name=self.name, create=False)

    def read(self, retries: int = 4) -> MarioFrame | None:
        """Return a complete new frame, or ``None`` during a concurrent write."""

        if retries <= 0:
            raise ValueError("retries must be positive")
        self.connect()
        assert self._shm is not None
        buf = self._shm.buf
        if len(buf) < FRAME_HEADER.size:
            raise RuntimeError("Mario framebuffer shared memory is too small")

        for _ in range(retries):
            first = FRAME_HEADER.unpack(bytes(buf[: FRAME_HEADER.size]))
            magic, width, height, channels, nbytes, sequence = first
            if magic != FRAME_MAGIC:
                raise RuntimeError("Mario framebuffer has an incompatible protocol")
            if channels != 3 or width <= 0 or height <= 0:
                raise RuntimeError("Mario framebuffer has invalid dimensions")
            if nbytes != width * height * channels:
                raise RuntimeError("Mario framebuffer byte count is inconsistent")
            if FRAME_HEADER.size + nbytes > len(buf):
                raise RuntimeError("Mario framebuffer payload exceeds shared memory")
            if sequence & 1:
                continue

            payload = bytes(buf[FRAME_HEADER.size : FRAME_HEADER.size + nbytes])
            second = FRAME_HEADER.unpack(bytes(buf[: FRAME_HEADER.size]))
            if first == second and not (second[-1] & 1):
                rgb = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
                return MarioFrame(sequence=sequence, rgb=rgb.copy())
        return None

    def close(self) -> None:
        if self._shm is not None:
            self._shm.close()
            self._shm = None

    def __enter__(self) -> "MarioFrameSubscriber":
        self.connect()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _resize_nearest(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    """Dependency-free nearest-neighbour resize for RGB display frames."""

    if rgb.shape[:2] == (height, width):
        return np.ascontiguousarray(rgb)
    ys = np.minimum(
        (np.arange(height, dtype=np.int64) * rgb.shape[0]) // height,
        rgb.shape[0] - 1,
    )
    xs = np.minimum(
        (np.arange(width, dtype=np.int64) * rgb.shape[1]) // width,
        rgb.shape[1] - 1,
    )
    return np.ascontiguousarray(rgb[ys[:, None], xs[None, :]])


class MarioMonitorTexture:
    """Copy RGB pixels into a compiled MuJoCo texture and upload them."""

    def __init__(
        self,
        model: mujoco.MjModel,
        texture_name: str = MARIO_TEXTURE_NAME,
    ) -> None:
        self.model = model
        self.texture_name = texture_name
        self.texture_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_TEXTURE, texture_name
        )
        if self.texture_id < 0:
            raise ValueError(f'MuJoCo texture "{texture_name}" does not exist')
        self.width = int(model.tex_width[self.texture_id])
        self.height = int(model.tex_height[self.texture_id])
        self._address = int(model.tex_adr[self.texture_id])

    def write(self, rgb: np.ndarray) -> None:
        """Update CPU-side ``tex_data`` from an HxWx3 uint8-compatible image."""

        frame = np.asarray(rgb)
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise ValueError("Mario frame must have shape (height, width, 3 or 4)")
        frame = np.asarray(frame[:, :, :3], dtype=np.uint8)
        frame = _resize_nearest(frame, self.height, self.width)
        size = self.width * self.height * 3
        self.model.tex_data[self._address : self._address + size] = frame.reshape(-1)

    def upload(self, renderer: mujoco.Renderer) -> None:
        """Upload the modified texture to this renderer's GPU context."""

        context = getattr(renderer, "_mjr_context", None)
        if context is None:
            raise RuntimeError("MuJoCo renderer has no active rendering context")
        gl_context = getattr(renderer, "_gl_context", None)
        if gl_context is not None:
            gl_context.make_current()
        mujoco.mjr_uploadTexture(self.model, context, self.texture_id)

    def update(self, rgb: np.ndarray, renderer: mujoco.Renderer) -> None:
        self.write(rgb)
        self.upload(renderer)


class MarioHeadCameraRenderer:
    """Render what Microduck sees after updating the in-world Mario display."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        width: int = 320,
        height: int = 240,
        camera_name: str = HEAD_CAMERA_NAME,
    ) -> None:
        self.model = model
        self.data = data
        self.camera_name = camera_name
        self.renderer = mujoco.Renderer(model, width=width, height=height)
        self.monitor = MarioMonitorTexture(model)

    def render(self, mario_rgb: np.ndarray) -> np.ndarray:
        """Upload Mario first, then capture Microduck's head-camera image."""

        self.monitor.update(mario_rgb, self.renderer)
        self.renderer.update_scene(self.data, camera=self.camera_name)
        return self.renderer.render()

    def render_latest(self, subscriber: MarioFrameSubscriber) -> MarioFrame | None:
        frame = subscriber.read()
        if frame is None:
            return None
        return MarioFrame(frame.sequence, self.render(frame.rgb))

    def close(self) -> None:
        self.renderer.close()

    def __enter__(self) -> "MarioHeadCameraRenderer":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
