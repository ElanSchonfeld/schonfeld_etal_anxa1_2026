"""Fail-closed resource checks for large local single-cell operations."""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
from pathlib import Path


GIB = 1024**3
DEFAULT_MAXIMUM_SWAP_USED_GIB = 4.0
SWAP_OVERRIDE_ENV = "FDA_ATLAS_MAXIMUM_SWAP_USED_GIB"
SWAP_OVERRIDE_REASON_ENV = "FDA_ATLAS_SWAP_OVERRIDE_REASON"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTERNAL_ROOT = Path(
    os.environ.get(
        "DOPABASE_EXTERNAL_ROOT",
        os.environ.get("DOPABASE_DATA_ROOT", PROJECT_ROOT / "data"),
    )
).expanduser()


def resource_snapshot(
    *,
    local_root: Path = Path.home(),
    external_root: Path = DEFAULT_EXTERNAL_ROOT,
) -> dict[str, float | bool]:
    external_root = Path(external_root)
    mounted = external_root.exists() and external_root.is_mount()
    local_free = shutil.disk_usage(local_root).free
    external_free = shutil.disk_usage(external_root).free if mounted else 0
    physical = int(
        subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    swap_text = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    match = re.search(r"used\s*=\s*([0-9.]+)([MG])", swap_text)
    if not match:
        raise RuntimeError(f"cannot parse swap usage: {swap_text.strip()}")
    scale = 1024**2 if match.group(2) == "M" else 1024**3
    swap_used = float(match.group(1)) * scale
    return {
        "external_mounted": mounted,
        "local_free_gib": local_free / GIB,
        "external_free_gib": external_free / GIB,
        "physical_memory_gib": physical / GIB,
        "swap_used_gib": swap_used / GIB,
    }


def validate_large_data_resources(
    snapshot: dict[str, float | bool],
    *,
    estimated_peak_memory_gib: float,
    minimum_local_free_gib: float = 20.0,
    minimum_external_free_gib: float = 100.0,
    maximum_swap_used_gib: float | None = None,
) -> None:
    """Reject a large job unless every explicit safety gate is satisfied."""

    failures = []
    if maximum_swap_used_gib is None:
        override_authorized = snapshot.get("swap_override_authorized", False)
        recorded_limit = snapshot.get(
            "maximum_swap_used_gib", DEFAULT_MAXIMUM_SWAP_USED_GIB
        )
        recorded_reason = str(snapshot.get("swap_override_reason", "")).strip()
        try:
            recorded_limit = float(recorded_limit)
        except (TypeError, ValueError):
            recorded_limit = math.nan
        if override_authorized is True:
            if not recorded_reason:
                failures.append("recorded swap override reason is missing")
            if (
                not math.isfinite(recorded_limit)
                or recorded_limit < DEFAULT_MAXIMUM_SWAP_USED_GIB
            ):
                failures.append("recorded swap override limit is invalid")
            maximum_swap_used_gib = recorded_limit
        else:
            if override_authorized not in (False, None):
                failures.append("recorded swap override authorization is invalid")
            if (
                "maximum_swap_used_gib" in snapshot
                and recorded_limit != DEFAULT_MAXIMUM_SWAP_USED_GIB
            ):
                failures.append("unauthorized recorded swap limit differs from default")
            maximum_swap_used_gib = DEFAULT_MAXIMUM_SWAP_USED_GIB
    if not snapshot.get("external_mounted", False):
        failures.append("external data root is not mounted")
    if float(snapshot.get("local_free_gib", 0)) < minimum_local_free_gib:
        failures.append(
            f"local free space is {float(snapshot.get('local_free_gib', 0)):.1f} GiB; "
            f"requires {minimum_local_free_gib:.1f} GiB"
        )
    if float(snapshot.get("external_free_gib", 0)) < minimum_external_free_gib:
        failures.append(
            f"external free space is {float(snapshot.get('external_free_gib', 0)):.1f} GiB; "
            f"requires {minimum_external_free_gib:.1f} GiB"
        )
    physical = float(snapshot.get("physical_memory_gib", 0))
    if estimated_peak_memory_gib <= 0 or estimated_peak_memory_gib >= physical * 0.5:
        failures.append(
            f"estimated peak memory {estimated_peak_memory_gib:.1f} GiB is not below "
            f"50 percent of {physical:.1f} GiB physical memory"
        )
    swap = float(snapshot.get("swap_used_gib", 0))
    if swap > maximum_swap_used_gib:
        failures.append(
            f"swap use is {swap:.1f} GiB; maximum safe preflight is "
            f"{maximum_swap_used_gib:.1f} GiB"
        )
    if failures:
        raise RuntimeError("large-data preflight failed: " + "; ".join(failures))


def large_data_preflight(
    *, estimated_peak_memory_gib: float
) -> dict[str, float | bool | str]:
    snapshot = resource_snapshot()
    maximum_swap_used_gib = DEFAULT_MAXIMUM_SWAP_USED_GIB
    override_value = os.environ.get(SWAP_OVERRIDE_ENV)
    override_reason = os.environ.get(SWAP_OVERRIDE_REASON_ENV, "").strip()
    override_authorized = override_value is not None
    if override_authorized:
        if not override_reason:
            raise RuntimeError(
                f"{SWAP_OVERRIDE_REASON_ENV} is required when {SWAP_OVERRIDE_ENV} is set"
            )
        try:
            maximum_swap_used_gib = float(override_value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{SWAP_OVERRIDE_ENV} must be numeric") from exc
        if (
            not math.isfinite(maximum_swap_used_gib)
            or maximum_swap_used_gib < DEFAULT_MAXIMUM_SWAP_USED_GIB
        ):
            raise RuntimeError(
                f"{SWAP_OVERRIDE_ENV} must be finite and at least "
                f"{DEFAULT_MAXIMUM_SWAP_USED_GIB:.1f}"
            )
    validate_large_data_resources(
        snapshot,
        estimated_peak_memory_gib=estimated_peak_memory_gib,
        maximum_swap_used_gib=maximum_swap_used_gib,
    )
    return {
        **snapshot,
        "maximum_swap_used_gib": maximum_swap_used_gib,
        "swap_override_authorized": override_authorized,
        "swap_override_reason": override_reason if override_authorized else "",
    }
