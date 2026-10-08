from gesture_client.gate import GestureGate


def test_gate_requires_stability_and_latches_until_gesture_clears():
    gate = GestureGate(threshold=0.6, stable_frames=3, cooldown_seconds=0)
    assert not gate.observe("thumbUp", 0.9, False, now=1)
    assert not gate.observe("thumbUp", 0.9, False, now=2)
    assert gate.observe("thumbUp", 0.9, False, now=3)
    assert not gate.observe("thumbUp", 0.9, False, now=4)
    assert not gate.observe(None, 0, False, now=5)
    assert not gate.observe("thumbUp", 0.9, False, now=6)


def test_gate_respects_confidence_busy_and_cooldown():
    gate = GestureGate(threshold=0.6, stable_frames=1, cooldown_seconds=3.5)
    assert not gate.observe("ok", 0.5, False, now=0)
    assert not gate.observe("ok", 0.9, True, now=1)
    assert gate.observe("ok", 0.9, False, now=2)
    gate.observe(None, 0, False, now=3)
    assert not gate.observe("heartSingle", 0.9, False, now=4)
    gate.observe(None, 0, False, now=5)
    assert gate.observe("heartSingle", 0.9, False, now=6)

