import pytest

from fda_atlas_v2.resource_safety import validate_large_data_resources
from fda_atlas_v2.resource_safety import large_data_preflight


def _safe_snapshot():
    return {
        "external_mounted": True,
        "local_free_gib": 30.0,
        "external_free_gib": 500.0,
        "physical_memory_gib": 64.0,
        "swap_used_gib": 1.0,
    }


def test_large_data_resource_gate_accepts_safe_snapshot():
    validate_large_data_resources(_safe_snapshot(), estimated_peak_memory_gib=8.0)


def test_large_data_resource_gate_reports_all_unsafe_conditions():
    snapshot = _safe_snapshot() | {
        "external_mounted": False,
        "local_free_gib": 14.0,
        "external_free_gib": 50.0,
        "swap_used_gib": 10.0,
    }
    with pytest.raises(RuntimeError) as error:
        validate_large_data_resources(snapshot, estimated_peak_memory_gib=40.0)
    message = str(error.value)
    assert "not mounted" in message
    assert "local free space" in message
    assert "external free space" in message
    assert "50 percent" in message
    assert "swap use" in message


def test_explicit_swap_override_is_validated_and_recorded(monkeypatch):
    snapshot = _safe_snapshot() | {"swap_used_gib": 10.0}
    monkeypatch.setattr(
        "fda_atlas_v2.resource_safety.resource_snapshot", lambda: snapshot
    )
    monkeypatch.setenv("FDA_ATLAS_MAXIMUM_SWAP_USED_GIB", "12")
    monkeypatch.setenv(
        "FDA_ATLAS_SWAP_OVERRIDE_REASON",
        "user explicitly authorized continued bounded execution",
    )

    observed = large_data_preflight(estimated_peak_memory_gib=8.0)

    assert observed["swap_used_gib"] == 10.0
    assert observed["maximum_swap_used_gib"] == 12.0
    assert observed["swap_override_authorized"] is True
    assert observed["swap_override_reason"].startswith("user explicitly")
    validate_large_data_resources(observed, estimated_peak_memory_gib=8.0)


def test_swap_override_requires_reason_and_keeps_other_gates(monkeypatch):
    snapshot = _safe_snapshot() | {"local_free_gib": 10.0, "swap_used_gib": 10.0}
    monkeypatch.setattr(
        "fda_atlas_v2.resource_safety.resource_snapshot", lambda: snapshot
    )
    monkeypatch.setenv("FDA_ATLAS_MAXIMUM_SWAP_USED_GIB", "12")
    with pytest.raises(RuntimeError, match="FDA_ATLAS_SWAP_OVERRIDE_REASON"):
        large_data_preflight(estimated_peak_memory_gib=8.0)

    monkeypatch.setenv("FDA_ATLAS_SWAP_OVERRIDE_REASON", "explicit user authorization")
    with pytest.raises(RuntimeError, match="local free space"):
        large_data_preflight(estimated_peak_memory_gib=8.0)


def test_recorded_swap_override_fails_closed_when_incomplete():
    snapshot = _safe_snapshot() | {
        "swap_used_gib": 10.0,
        "maximum_swap_used_gib": 12.0,
        "swap_override_authorized": True,
        "swap_override_reason": "",
    }
    with pytest.raises(RuntimeError, match="override reason"):
        validate_large_data_resources(snapshot, estimated_peak_memory_gib=8.0)

    snapshot["swap_override_authorized"] = False
    with pytest.raises(RuntimeError, match="unauthorized recorded swap limit"):
        validate_large_data_resources(snapshot, estimated_peak_memory_gib=8.0)
