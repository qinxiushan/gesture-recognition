import cv2
import numpy as np
import pytest

from gesture_client.mjpeg import MjpegParser


def part(jpeg, sequence=1, length=True, boundary=b'camera-test'):
    return (b'--' + boundary + b'\r\nContent-Type: image/jpeg\r\n' +
            (b'Content-Length: ' + str(len(jpeg)).encode() + b'\r\n' if length else b'') +
            b'X-Frame-Sequence: ' + str(sequence).encode() + b'\r\n\r\n' + jpeg + b'\r\n')


@pytest.mark.parametrize('chunk_size', [1, 2, 7, 64, 100_000])
def test_multipart_handles_split_headers_markers_and_multiple_frames(chunk_size):
    _, encoded = cv2.imencode('.jpg', np.zeros((48, 64, 3), np.uint8))
    jpeg = encoded.tobytes()
    body = b'preamble\r\n' + part(jpeg, 10) + part(jpeg, 11) + b'--camera-test--\r\n'
    parser = MjpegParser('multipart/x-mixed-replace; boundary="camera-test"')
    packets = []
    for index in range(0, len(body), chunk_size):
        packets.extend(parser.feed(body[index:index + chunk_size]))
    assert [p.sequence for p in packets] == [10, 11]
    assert all(p.jpeg == jpeg for p in packets)
    assert cv2.imdecode(np.frombuffer(packets[0].jpeg, np.uint8), cv2.IMREAD_COLOR).shape == (48, 64, 3)
    assert parser.closed and not parser.buffer


def test_content_length_ignores_embedded_jpeg_markers():
    jpeg = b'\xff\xd8metadata\xff\xd8thumbnail\xff\xd9real-image\xff\xd9'
    parser = MjpegParser('multipart/x-mixed-replace; boundary=camera-test')
    assert parser.feed(part(jpeg))[0].jpeg == jpeg


def test_without_content_length_waits_for_next_boundary():
    jpeg = b'\xff\xd8image\xff\xd9'
    parser = MjpegParser('multipart/x-mixed-replace; boundary=camera-test')
    assert parser.feed(part(jpeg, length=False)) == []
    assert parser.feed(b'--camera-test--\r\n')[0].jpeg == jpeg


def test_unbounded_or_invalid_parts_fail_without_growing_forever():
    parser = MjpegParser('multipart/x-mixed-replace; boundary=camera-test', max_frame_bytes=32)
    with pytest.raises(ValueError, match='length'):
        parser.feed(part(b'\xff\xd8' + b'x' * 100 + b'\xff\xd9'))
    parser = MjpegParser('multipart/x-mixed-replace; boundary=camera-test', max_frame_bytes=32)
    with pytest.raises(ValueError, match='large'):
        parser.feed(part(b'\xff\xd8' + b'x' * 100, length=False))
    parser = MjpegParser('multipart/x-mixed-replace; boundary=camera-test')
    parser.feed(b'preamble' * 10_000)
    assert len(parser.buffer) <= len(parser.delimiter) + 2
