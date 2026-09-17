"""Dependency-free UDP bridge from Microduck pad levels to Mario sidecar."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import socket
import time

from mjlab_microduck.controller_game import ControllerFrame


PROTOCOL_VERSION = 1
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 55355
DEFAULT_REQUEST_PORT = 55356


def encode_controller_packet(frame: ControllerFrame, sequence: int) -> bytes:
    """Serialize one measured controller frame for the emulator sidecar."""

    if sequence < 0:
        raise ValueError("sequence must be non-negative")
    return json.dumps(
        {
            "v": PROTOCOL_VERSION,
            "seq": sequence,
            "left": frame.left,
            "right": frame.right,
            "jump": frame.jump,
        },
        separators=(",", ":"),
    ).encode("ascii")


def decode_controller_packet(payload: bytes) -> tuple[int, ControllerFrame]:
    """Validate a controller packet received by the Python 3.13 sidecar."""

    message = json.loads(payload.decode("ascii"))
    if message.get("v") != PROTOCOL_VERSION:
        raise ValueError("unsupported controller protocol version")
    sequence = message.get("seq")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
        raise ValueError("invalid controller sequence")
    levels = []
    for key in ("left", "right", "jump"):
        value = message.get(key)
        if not isinstance(value, bool):
            raise ValueError(f"controller field {key!r} must be boolean")
        levels.append(value)
    return sequence, ControllerFrame(*levels)


@dataclass(slots=True)
class SuperMarioUdpClient:
    """Send measured physical-pad levels to the local emulator process."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    _sequence: int = field(default=0, init=False)
    _socket: socket.socket = field(
        default_factory=lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM),
        init=False,
    )

    def send(self, frame: ControllerFrame) -> None:
        payload = encode_controller_packet(frame, self._sequence)
        self._socket.sendto(payload, (self.host, self.port))
        self._sequence += 1

    def close(self) -> None:
        self._socket.close()

    def __enter__(self) -> "SuperMarioUdpClient":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


@dataclass(slots=True)
class FlybrainUdpReceiver:
    """Receive high-level flybrain requests with sequence and deadman guards.

    The returned frame belongs in the Mario PPO policy's three twist-command
    slots. It must never be forwarded directly to the emulator.
    """

    host: str = DEFAULT_HOST
    port: int = DEFAULT_REQUEST_PORT
    timeout_s: float = 0.5
    _socket: socket.socket = field(init=False)
    _frame: ControllerFrame = field(default_factory=ControllerFrame, init=False)
    _last_sequence: int = field(default=-1, init=False)
    _last_received: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if self.timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((self.host, self.port))
        self._socket.setblocking(False)

    def poll(self) -> ControllerFrame:
        while True:
            try:
                payload, _ = self._socket.recvfrom(4096)
            except BlockingIOError:
                break
            try:
                sequence, frame = decode_controller_packet(payload)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                continue
            if sequence > self._last_sequence:
                self._last_sequence = sequence
                self._frame = frame
                self._last_received = time.monotonic()
        if time.monotonic() - self._last_received > self.timeout_s:
            return ControllerFrame()
        return self._frame

    @property
    def command_vector(self) -> tuple[float, float, float]:
        frame = self.poll()
        return float(frame.left), float(frame.right), float(frame.jump)

    def close(self) -> None:
        self._socket.close()

    def __enter__(self) -> "FlybrainUdpReceiver":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
