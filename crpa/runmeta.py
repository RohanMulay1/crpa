"""
crpa.runmeta - run identity, provenance metadata, and atomic result writes.

Every result file carries enough provenance to be audited later, and an
explicit :class:`Status`. The point of the status field is that a run which
never executed, or died of OOM, must never be readable as a number.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch


class Status(str, Enum):
    """Execution status of an experiment record.

    ``COMPLETED``   ran to completion at the configured scale
    ``SMOKE``       ran at reduced scale to validate the code path only
    ``NOT_RUN``     implemented but never executed
    ``OOM``         attempted and ran out of memory - never a numeric result
    ``UNSUPPORTED`` the hardware or the software stack cannot run it
    ``FAILED``      attempted and raised
    """

    COMPLETED = "completed"
    SMOKE = "smoke"
    NOT_RUN = "not_run"
    OOM = "oom"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"

    @property
    def is_numeric(self) -> bool:
        """Whether metrics attached to this record may be read as measurements."""
        return self in (Status.COMPLETED, Status.SMOKE)


def git_sha(repo: Optional[Path] = None) -> str:
    """Current commit SHA, or ``"unknown"`` outside a git checkout."""
    repo = repo or Path(__file__).resolve().parent.parent
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def git_dirty(repo: Optional[Path] = None) -> bool:
    """Whether the working tree has uncommitted changes."""
    repo = repo or Path(__file__).resolve().parent.parent
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return out.returncode == 0 and bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def gpu_info() -> Dict[str, Any]:
    """GPU model, capability and memory, or an explicit CPU-only record."""
    if not torch.cuda.is_available():
        return {
            "available": False,
            "name": None,
            "count": 0,
            "total_memory_gb": None,
            "capability": None,
        }
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "count": torch.cuda.device_count(),
        "total_memory_gb": round(props.total_memory / 1e9, 2),
        "capability": "{}.{}".format(props.major, props.minor),
    }


def environment() -> Dict[str, Any]:
    """Everything a reader needs to judge whether a number is comparable."""
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu": gpu_info(),
        "cpu_count": os.cpu_count(),
    }


def run_id(config_json: str, seed: int, extra: str = "") -> str:
    """Deterministic 12-hex-char id derived from configuration and seed.

    The same configuration always yields the same id, which is what makes
    resumption and skip-if-done possible without a database.
    """
    payload = "{}|seed={}|{}".format(config_json, seed, extra)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass
class RunRecord:
    """One experiment result, with provenance and an explicit status."""

    run_id: str
    experiment: str
    status: Status
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    context_length: Optional[int] = None
    dtype: Optional[str] = None
    variant: Optional[str] = None
    splits: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    error: Optional[str] = None
    duration_s: Optional[float] = None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    git_sha: str = field(default_factory=git_sha)
    git_dirty: bool = field(default_factory=git_dirty)
    env: Dict[str, Any] = field(default_factory=environment)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunRecord":
        d = dict(d)
        d["status"] = Status(d["status"])
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def require_numeric(self) -> None:
        """Raise unless this record's metrics may be treated as measurements."""
        if not self.status.is_numeric:
            raise ValueError(
                "run {} has status {!r}; its metrics are not measurements and "
                "must not be plotted or tabulated".format(self.run_id, self.status.value)
            )


def atomic_write_text(path: "str | Path", text: str) -> Path:
    """Write via a temporary file plus ``os.replace`` so readers never see a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def write_json(path: "str | Path", payload: Any) -> Path:
    """Atomically write pretty-printed JSON."""
    return atomic_write_text(path, json.dumps(payload, indent=2, default=str))


def read_json(path: "str | Path") -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("expected results file not found: {}".format(path))
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_csv(path: "str | Path", rows: Sequence[Dict[str, Any]],
              fieldnames: Optional[Sequence[str]] = None) -> Path:
    """Atomically write rows as CSV.

    Fails loudly on empty input rather than leaving a misleading empty file.
    """
    if not rows:
        raise ValueError("refusing to write an empty CSV to {}".format(path))
    if fieldnames is None:
        seen: List[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(fieldnames), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return atomic_write_text(path, buf.getvalue())


def save_record(results_dir: "str | Path", record: RunRecord) -> Path:
    """Persist one :class:`RunRecord` to ``<results_dir>/runs/<run_id>.json``."""
    path = Path(results_dir) / "runs" / "{}.json".format(record.run_id)
    return write_json(path, record.to_dict())


def load_record(results_dir: "str | Path", rid: str) -> Optional[RunRecord]:
    """Load a record by id, or ``None`` if it has not been produced yet."""
    path = Path(results_dir) / "runs" / "{}.json".format(rid)
    if not path.exists():
        return None
    return RunRecord.from_dict(read_json(path))


def load_records(results_dir: "str | Path") -> List[RunRecord]:
    """Load every record under ``<results_dir>/runs``, sorted by id."""
    runs = Path(results_dir) / "runs"
    if not runs.exists():
        return []
    out: List[RunRecord] = []
    for path in sorted(runs.glob("*.json")):
        try:
            out.append(RunRecord.from_dict(read_json(path)))
        except (KeyError, ValueError) as exc:
            raise ValueError("malformed run record at {}: {}".format(path, exc)) from exc
    return out


def numeric_records(results_dir: "str | Path", experiment: Optional[str] = None) -> List[RunRecord]:
    """Load only records whose metrics are real measurements.

    This is the accessor plotting and aggregation code must use, so a
    ``not_run`` or ``oom`` record can never leak into a figure as a number.
    """
    records = load_records(results_dir)
    if experiment is not None:
        records = [r for r in records if r.experiment == experiment]
    return [r for r in records if r.status.is_numeric]


def is_complete(results_dir: "str | Path", rid: str, force: bool = False) -> bool:
    """Whether a finished record already exists, for skip-and-resume.

    ``force=True`` always reports incomplete. Non-numeric statuses (a previous
    OOM or failure) also report incomplete so a retry is possible.
    """
    if force:
        return False
    record = load_record(results_dir, rid)
    return record is not None and record.status.is_numeric


def summarise_statuses(results_dir: "str | Path") -> Dict[str, int]:
    """Count records by status - used by the CLI to report what actually ran."""
    counts: Dict[str, int] = {}
    for record in load_records(results_dir):
        counts[record.status.value] = counts.get(record.status.value, 0) + 1
    return counts
