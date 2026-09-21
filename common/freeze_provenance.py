#!/usr/bin/env python3
"""Staleness guard for frozen figure data."""
import hashlib
import json
import os
from pathlib import Path

STAMP_NAME = "_provenance.json"
REPO_ROOT = Path(__file__).resolve().parents[1]
PATH_ROOTS = (
    "PPMI_ROOT",
    "M2H_SOURCE_ROOT",
    "DOPABASE_DATA_ROOT",
    "DOPABASE_SOURCE_ROOT",
)


def _sha(path):
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _logical_path(path):
    """Return a portable repository- or environment-rooted input identifier."""
    p = Path(path).expanduser().resolve()
    try:
        return "${REPO_ROOT}/" + p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        pass
    for variable in PATH_ROOTS:
        value = os.environ.get(variable)
        if not value:
            continue
        root = Path(value).expanduser().resolve()
        try:
            return "${" + variable + "}/" + p.relative_to(root).as_posix()
        except ValueError:
            continue
    raise ValueError(
        f"Cannot record a portable provenance path for {p}. Place it under the "
        f"repository or one of: {', '.join(PATH_ROOTS)}"
    )


def _resolve_logical_path(identifier):
    prefix = "${REPO_ROOT}/"
    if identifier.startswith(prefix):
        return REPO_ROOT / identifier[len(prefix):], None
    for variable in PATH_ROOTS:
        prefix = "${" + variable + "}/"
        if identifier.startswith(prefix):
            value = os.environ.get(variable)
            if not value:
                return None, f"{variable} is not set"
            return Path(value).expanduser().resolve() / identifier[len(prefix):], None
    path = Path(identifier)
    if path.is_absolute():
        return path, None
    return REPO_ROOT / path, None


def stamp(frozen_dir, tag, upstream_paths):
    """Record content hashes of the upstream files this freeze consumed, under `tag`."""
    frozen_dir = Path(frozen_dir)
    f = frozen_dir / STAMP_NAME
    prov = json.load(open(f)) if f.exists() else {}
    portable = {}
    for old_tag, files in prov.items():
        portable[old_tag] = {
            (_logical_path(path) if Path(path).is_absolute() else path): digest
            for path, digest in files.items()
        }
    prov = portable
    prov[tag] = {_logical_path(p): _sha(p) for p in upstream_paths}
    frozen_dir.mkdir(parents=True, exist_ok=True)
    json.dump(prov, open(f, "w"), indent=1, sort_keys=True)
    n_missing = sum(1 for v in prov[tag].values() if v is None)
    print(f"  [provenance] stamped {tag}: {len(prov[tag])} upstream file(s)"
          + (f", {n_missing} MISSING" if n_missing else ""))


def check(frozen_dir, verbose=True):
    """Compare recorded hashes against the upstream files as they are now."""
    f = Path(frozen_dir) / STAMP_NAME
    if not f.exists():
        if verbose:
            print(f"  [provenance] no {STAMP_NAME} -- freshness UNKNOWN (re-run the compute/freeze_*.py scripts to stamp it)")
        return [("*", str(f), "no provenance stamp")]
    prov = json.load(open(f))
    stale = []
    for tag, files in prov.items():
        for path, rec in files.items():
            resolved, resolution_error = _resolve_logical_path(path)
            if resolution_error:
                stale.append((tag, path, resolution_error))
                continue
            cur = _sha(resolved)
            if cur is None:
                stale.append((tag, path, "upstream file MISSING"))
            elif rec is None:
                stale.append((tag, path, "was missing at freeze time"))
            elif cur != rec:
                stale.append((tag, path, f"changed since freeze ({rec} -> {cur})"))
    if verbose:
        if stale:
            print("\n" + "!" * 78)
            print("!! STALE FROZEN DATA -- these panels were built from tables that predate their source")
            for tag, path, why in stale:
                print(f"!!   [{tag}] {Path(path).name}: {why}")
            print("!! Re-run the matching compute/freeze_*.py, then the code/panel_*.py, then assemble.")
            print("!" * 78 + "\n")
        else:
            print(f"  [provenance] frozen data is current ({sum(len(v) for v in prov.values())} upstream files checked)")
    return stale
