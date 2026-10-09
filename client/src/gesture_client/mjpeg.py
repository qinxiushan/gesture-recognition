"""Incremental, bounded multipart parser; network chunks are not frames."""
from __future__ import annotations

from dataclasses import dataclass
from email.message import Message


@dataclass(frozen=True)
class JpegPacket:
    jpeg: bytes
    sequence: int | None = None
    captured_at: float | None = None


class MjpegParser:
    def __init__(self, content_type: str, max_frame_bytes: int = 4 * 1024 * 1024):
        message = Message()
        message['Content-Type'] = content_type
        boundary = message.get_param('boundary')
        if message.get_content_type() != 'multipart/x-mixed-replace' or not boundary:
            raise ValueError('Expected multipart MJPEG with a boundary')
        if len(boundary) > 70 or '\r' in boundary or '\n' in boundary:
            raise ValueError('Invalid MJPEG boundary')
        self.delimiter = b'--' + boundary.encode('ascii')
        self.buffer = bytearray()
        self.max_frame_bytes = max_frame_bytes
        self.headers: dict[str, str] | None = None
        self.length: int | None = None
        self.closed = False

    def feed(self, chunk: bytes) -> list[JpegPacket]:
        if self.closed:
            return []
        self.buffer.extend(chunk)
        packets = []
        while True:
            if self.headers is None:
                # Locate a boundary line, never an arbitrary occurrence in a JPEG.
                if self.buffer.startswith(self.delimiter):
                    start = 0
                else:
                    index = self.buffer.find(b'\r\n' + self.delimiter)
                    start = index + 2 if index >= 0 else -1
                if start < 0:
                    # Keep the possible split delimiter suffix, discard preamble.
                    if len(self.buffer) > 8192:
                        del self.buffer[:-len(self.delimiter) - 2]
                    break
                if start:
                    del self.buffer[:start]
                line_end = self.buffer.find(b'\r\n')
                if line_end < 0:
                    if len(self.buffer) > 8192:
                        raise ValueError('MJPEG boundary line too large')
                    break
                line = bytes(self.buffer[:line_end]).rstrip(b' \t')
                if line == self.delimiter + b'--':
                    self.closed = True
                    self.buffer.clear()
                    break
                if line != self.delimiter:
                    del self.buffer[:line_end + 2]
                    continue
                header_end = self.buffer.find(b'\r\n\r\n', line_end)
                if header_end < 0:
                    if len(self.buffer) > 8192:
                        raise ValueError('MJPEG headers too large')
                    break
                if header_end > 8192:
                    raise ValueError('MJPEG headers too large')
                headers = {}
                for header in bytes(self.buffer[line_end + 2:header_end]).split(b'\r\n'):
                    name, separator, value = header.partition(b':')
                    if not separator:
                        raise ValueError('Malformed MJPEG header')
                    headers[name.decode('ascii').lower()] = value.decode('ascii').strip()
                if headers.get('content-type', '').lower() != 'image/jpeg':
                    raise ValueError('Expected JPEG part')
                length = headers.get('content-length')
                self.length = int(length) if length is not None else None
                if self.length is not None and not 0 < self.length <= self.max_frame_bytes:
                    raise ValueError('Invalid MJPEG frame length')
                self.headers = headers
                del self.buffer[:header_end + 4]

            if self.length is None:
                end = self.buffer.find(b'\r\n' + self.delimiter)
                if end < 0:
                    if len(self.buffer) > self.max_frame_bytes + len(self.delimiter) + 2:
                        raise ValueError('MJPEG frame too large')
                    break
                if end > self.max_frame_bytes:
                    raise ValueError('MJPEG frame too large')
            else:
                end = self.length
                if len(self.buffer) < end:
                    break
            jpeg = bytes(self.buffer[:end])
            del self.buffer[:end]
            headers, self.headers = self.headers, None
            self.length = None
            if not (jpeg.startswith(b'\xff\xd8') and jpeg.endswith(b'\xff\xd9')):
                raise ValueError('Incomplete JPEG part')
            packets.append(JpegPacket(
                jpeg,
                int(headers['x-frame-sequence']) if 'x-frame-sequence' in headers else None,
                float(headers['x-captured-at']) if 'x-captured-at' in headers else None))
        return packets
