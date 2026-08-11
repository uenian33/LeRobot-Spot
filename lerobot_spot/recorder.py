"""Episode recording for hand-guided demonstrations.

Writes two things per episode, on purpose:

* `episode_XXXX.jsonl`, one line per control tick, flushed as it goes. This is
  the crash-proof copy. A collection session is a person standing next to a
  powered robot for an hour, and losing that hour to a stray exception at tick
  50,000 is not acceptable.
* `episode_XXXX.npz`, written when the episode closes -- every field stacked
  into one array of shape `(ticks, ...)`. This is the copy you actually train
  from; `load_episode` gives it back as a dict of arrays.

Nothing here decides what a sample contains. The caller passes a flat dict of
scalars and numpy arrays and the recorder stacks whatever it is given, so adding
a field to the collection loop needs no change in this file.

Discarded episodes are deleted outright, both files. Demonstration data is only
worth what its worst episode is worth, and an operator who has to filter bad
takes later will not do it.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

LOGGER = logging.getLogger(__name__)


def _jsonable(value):
    """Flatten one sample value into something `json.dumps` accepts."""
    if isinstance(value, np.ndarray):
        return [None if not np.isfinite(v) else float(v) for v in value.reshape(-1)]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, str):
        return value
    return float(value)


@dataclass
class EpisodeSummary:
    """What the UI needs to show about an episode after it closes."""

    index: int
    ticks: int
    duration: float
    path: Optional[Path]
    kept: bool


class EpisodeRecorder:
    """Owns a session directory and the episode currently being written."""

    def __init__(self, root: Path, metadata: Optional[dict] = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata or {})

        self._index = self._next_index()
        self._handle = None
        self._samples: list[dict] = []
        self._started_at = 0.0
        self._episode_dir: Optional[Path] = None

        (self.root / "session.json").write_text(
            json.dumps({"created": time.time(), **self.metadata}, indent=2, default=str) + "\n"
        )

    def _next_index(self) -> int:
        """Resume numbering after existing episodes, so a session can be re-entered."""
        existing = [p.name for p in self.root.glob("episode_*")]
        indices = []
        for name in existing:
            stem = name.split(".")[0]
            try:
                indices.append(int(stem.split("_")[1]))
            except (IndexError, ValueError):
                continue
        return max(indices) + 1 if indices else 0

    # -- state --------------------------------------------------------------

    @property
    def recording(self) -> bool:
        return self._handle is not None

    @property
    def index(self) -> int:
        return self._index

    @property
    def ticks(self) -> int:
        return len(self._samples)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started_at if self.recording else 0.0

    # -- lifecycle ----------------------------------------------------------

    def start(self, task: str = "") -> int:
        """Open a new episode. Returns its index."""
        if self.recording:
            raise RuntimeError("an episode is already recording")
        self._episode_dir = self.root / f"episode_{self._index:04d}"
        self._episode_dir.mkdir(parents=True, exist_ok=True)
        self._handle = (self._episode_dir / "samples.jsonl").open("w", buffering=1)
        self._samples = []
        self._started_at = time.monotonic()
        (self._episode_dir / "meta.json").write_text(
            json.dumps(
                {
                    "index": self._index,
                    "task": task,
                    "started": time.time(),
                    **self.metadata,
                },
                indent=2,
                default=str,
            )
            + "\n"
        )
        return self._index

    def append(self, sample: dict) -> None:
        """Record one control tick. Silently ignored when not recording."""
        if not self.recording:
            return
        self._samples.append(sample)
        try:
            self._handle.write(json.dumps({k: _jsonable(v) for k, v in sample.items()}) + "\n")
        except (OSError, TypeError) as err:
            # A write failure must not take the control loop down with it -- the
            # operator is holding a powered arm.
            LOGGER.error("Dropped a sample: %s", err)

    def stop(self, keep: bool = True, task: str = "") -> EpisodeSummary:
        """Close the episode. `keep=False` deletes it entirely."""
        if not self.recording:
            raise RuntimeError("no episode is recording")

        duration = self.elapsed
        ticks = len(self._samples)
        self._handle.close()
        self._handle = None
        episode_dir = self._episode_dir
        self._episode_dir = None

        if not keep or ticks == 0:
            shutil.rmtree(episode_dir, ignore_errors=True)
            self._samples = []
            return EpisodeSummary(self._index, ticks, duration, None, kept=False)

        path = episode_dir / "samples.npz"
        try:
            np.savez_compressed(path, **self._stack())
        except (OSError, ValueError) as err:
            # The jsonl is already on disk and holds everything, so this is a
            # degraded save, not a lost episode.
            LOGGER.error("Could not write %s (jsonl is intact): %s", path, err)
            path = episode_dir / "samples.jsonl"

        (episode_dir / "meta.json").write_text(
            json.dumps(
                {
                    "index": self._index,
                    "task": task,
                    "ended": time.time(),
                    "ticks": ticks,
                    "duration": duration,
                    **self.metadata,
                },
                indent=2,
                default=str,
            )
            + "\n"
        )

        summary = EpisodeSummary(self._index, ticks, duration, path, kept=True)
        self._index += 1
        self._samples = []
        return summary

    def close(self) -> Optional[EpisodeSummary]:
        """Flush anything in flight. Keeps the episode -- an interrupted take is data."""
        return self.stop(keep=True) if self.recording else None

    # -- internals ----------------------------------------------------------

    def _stack(self) -> dict:
        """Stack samples into `(ticks, ...)` arrays, one per field.

        Fields that appear in some ticks but not others are padded with NaN, so
        an intermittently-missing frame transform cannot silently shorten or
        misalign a column.
        """
        keys: list = []
        for sample in self._samples:
            for key in sample:
                if key not in keys:
                    keys.append(key)

        widths = {}
        for key in keys:
            for sample in self._samples:
                value = sample.get(key)
                if isinstance(value, np.ndarray):
                    widths[key] = value.reshape(-1).shape[0]
                    break

        out = {}
        for key in keys:
            width = widths.get(key)
            rows = []
            for sample in self._samples:
                value = sample.get(key)
                if width is None:
                    rows.append(np.nan if value is None else value)
                elif isinstance(value, np.ndarray):
                    rows.append(value.reshape(-1).astype(float))
                else:
                    rows.append(np.full(width, np.nan))
            out[key] = np.asarray(rows)
        return out


def load_episode(path: Path) -> dict:
    """Read one episode back as a dict of `(ticks, ...)` arrays.

    Accepts the episode directory, the `.npz`, or the `.jsonl` -- the last so a
    session interrupted mid-episode is still loadable.
    """
    path = Path(path)
    if path.is_dir():
        npz, jsonl = path / "samples.npz", path / "samples.jsonl"
        path = npz if npz.exists() else jsonl

    if path.suffix == ".npz":
        with np.load(path) as data:
            return {key: data[key] for key in data.files}

    samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not samples:
        return {}

    keys: list = []
    for sample in samples:
        for key in sample:
            if key not in keys:
                keys.append(key)

    out = {}
    for key in keys:
        column = [sample.get(key) for sample in samples]
        try:
            # `None` is how the writer spells a non-finite or absent value.
            out[key] = np.array(
                [np.nan if v is None else v for v in column], dtype=float
            )
        except (TypeError, ValueError):
            # Ragged or non-numeric (a task string, say); hand it back as-is.
            out[key] = np.array(column, dtype=object)
    return out
