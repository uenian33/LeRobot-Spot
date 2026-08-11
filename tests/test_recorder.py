"""Tests for episode recording.

The properties that matter for a data-collection rig: nothing is lost when the
process dies mid-take, discarded takes leave nothing behind, and what comes back
out of `load_episode` is aligned with what went in.
"""

import json

import numpy as np
import pytest

from lerobot_spot.recorder import EpisodeRecorder, load_episode


def sample(t: float, **overrides) -> dict:
    out = {
        "t": t,
        "joint_position": np.arange(6, dtype=float) + t,
        "hand_in_odom": np.arange(7, dtype=float),
        "gripper_open_percentage": 50.0,
        "engaged": True,
    }
    out.update(overrides)
    return out


def test_episode_roundtrips_through_npz(tmp_path):
    recorder = EpisodeRecorder(tmp_path, metadata={"task": "pick"})
    recorder.start(task="pick")
    for i in range(5):
        recorder.append(sample(float(i)))
    summary = recorder.stop(keep=True, task="pick")

    assert summary.kept and summary.ticks == 5
    data = load_episode(summary.path)
    assert data["joint_position"].shape == (5, 6)
    assert data["hand_in_odom"].shape == (5, 7)
    assert np.allclose(data["t"], [0, 1, 2, 3, 4])
    assert np.allclose(data["joint_position"][2], np.arange(6) + 2)


def test_jsonl_is_written_as_it_goes_so_a_crash_loses_nothing(tmp_path):
    """The npz only appears at stop(); the jsonl must be complete before then."""
    recorder = EpisodeRecorder(tmp_path)
    index = recorder.start()
    for i in range(3):
        recorder.append(sample(float(i)))

    # Simulate the process dying here: read the jsonl without ever calling stop().
    path = tmp_path / f"episode_{index:04d}" / "samples.jsonl"
    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(lines) == 3
    assert lines[2]["t"] == 2.0
    assert lines[0]["joint_position"] == list(range(6))


def test_load_episode_reads_an_interrupted_jsonl(tmp_path):
    recorder = EpisodeRecorder(tmp_path)
    index = recorder.start()
    for i in range(4):
        recorder.append(sample(float(i)))

    data = load_episode(tmp_path / f"episode_{index:04d}" / "samples.jsonl")
    assert data["joint_position"].shape == (4, 6)
    assert np.allclose(data["t"], [0, 1, 2, 3])


def test_load_episode_accepts_the_directory(tmp_path):
    recorder = EpisodeRecorder(tmp_path)
    recorder.start()
    recorder.append(sample(0.0))
    summary = recorder.stop()
    assert load_episode(summary.path.parent)["t"].shape == (1,)


def test_discarding_removes_the_episode_entirely(tmp_path):
    recorder = EpisodeRecorder(tmp_path)
    index = recorder.start()
    recorder.append(sample(0.0))
    summary = recorder.stop(keep=False)

    assert not summary.kept
    assert not (tmp_path / f"episode_{index:04d}").exists()
    # The index is reused, so a discarded take leaves no gap in the numbering.
    assert recorder.start() == index


def test_empty_episodes_are_dropped_rather_than_saved(tmp_path):
    recorder = EpisodeRecorder(tmp_path)
    recorder.start()
    summary = recorder.stop(keep=True)
    assert not summary.kept
    assert list(tmp_path.glob("episode_*")) == []


def test_indices_increment_and_resume_across_sessions(tmp_path):
    first = EpisodeRecorder(tmp_path)
    for _ in range(2):
        first.start()
        first.append(sample(0.0))
        first.stop()

    # A fresh recorder on the same directory must not overwrite episode 0 and 1.
    resumed = EpisodeRecorder(tmp_path)
    assert resumed.start() == 2


def test_missing_fields_are_padded_rather_than_misaligned(tmp_path):
    """A dropped frame transform must not shift a column by one tick."""
    recorder = EpisodeRecorder(tmp_path)
    recorder.start()
    recorder.append(sample(0.0))
    recorder.append({"t": 1.0})  # everything else missing this tick
    recorder.append(sample(2.0))
    summary = recorder.stop()

    data = load_episode(summary.path)
    assert data["joint_position"].shape == (3, 6)
    assert np.all(np.isnan(data["joint_position"][1]))
    assert np.allclose(data["joint_position"][2], np.arange(6) + 2)


def test_non_finite_values_survive_the_jsonl_as_nan(tmp_path):
    recorder = EpisodeRecorder(tmp_path)
    index = recorder.start()
    recorder.append(sample(0.0, deflection=np.full(7, np.nan)))
    recorder.stop()

    data = load_episode(tmp_path / f"episode_{index:04d}" / "samples.jsonl")
    assert np.all(np.isnan(data["deflection"][0]))


def test_append_outside_an_episode_is_ignored(tmp_path):
    recorder = EpisodeRecorder(tmp_path)
    recorder.append(sample(0.0))  # must not raise
    assert recorder.ticks == 0


def test_double_start_is_an_error(tmp_path):
    recorder = EpisodeRecorder(tmp_path)
    recorder.start()
    with pytest.raises(RuntimeError):
        recorder.start()


def test_stop_without_start_is_an_error(tmp_path):
    with pytest.raises(RuntimeError):
        EpisodeRecorder(tmp_path).stop()


def test_close_keeps_an_interrupted_take(tmp_path):
    recorder = EpisodeRecorder(tmp_path)
    recorder.start()
    recorder.append(sample(0.0))
    summary = recorder.close()
    assert summary is not None and summary.kept
    assert recorder.close() is None


def test_metadata_lands_in_session_and_episode_files(tmp_path):
    recorder = EpisodeRecorder(tmp_path, metadata={"control_frame": "odom", "rate_hz": 30.0})
    index = recorder.start(task="wipe the table")
    recorder.append(sample(0.0))
    recorder.stop(task="wipe the table")

    session = json.loads((tmp_path / "session.json").read_text())
    assert session["control_frame"] == "odom"

    meta = json.loads((tmp_path / f"episode_{index:04d}" / "meta.json").read_text())
    assert meta["task"] == "wipe the table"
    assert meta["ticks"] == 1
    assert meta["rate_hz"] == 30.0
