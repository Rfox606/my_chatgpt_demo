from adaptive_awr_v11.adapter_state_machine import AdapterStateMachine
from adaptive_awr_v11.config import AdaptiveAWRV11Config


def test_continuous_freeze_reason_creates_one_enter_episode() -> None:
    machine = AdapterStateMachine(AdaptiveAWRV11Config())
    for window in range(100):
        machine.request_freeze(window, "HIGH_RISK", {"risk": 1.0})
    enters = [event for event in machine.events if event["event_type"] == "ENTER_STATE"]
    assert len(enters) == 1
    assert not any(event["event_type"] == "FREEZE" for event in machine.events)
