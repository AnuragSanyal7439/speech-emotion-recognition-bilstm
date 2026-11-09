from pathlib import Path

import pandas as pd
import pytest

from config import EMOTIONS
from src.dataset import DatasetError, inspect_dataset, parse_ravdess_filename, split_by_actor


def test_ravdess_label_order_and_mapping():
    assert EMOTIONS == (
        "Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"
    )
    for code, name in enumerate(EMOTIONS, start=1):
        filename = f"03-01-{code:02d}-01-02-01-24.wav"
        clip = parse_ravdess_filename(Path("Actor_24") / filename)
        assert (clip.actor, clip.label, clip.emotion) == (24, code - 1, name)
        assert (clip.intensity, clip.statement, clip.repetition) == (1, 2, 1)
        assert clip.filename == filename


@pytest.mark.parametrize("filename", [
    "speech.wav",
    "01-01-06-01-02-01-24.wav",
    "03-02-06-01-02-01-24.wav",
    "03-01-09-01-02-01-24.wav",
    "03-01-01-02-02-01-24.wav",
    "03-01-02-01-02-01-25.wav",
    "03-01-02-01-02-01-24.mp3",
])
def test_rejects_invalid_speech_filenames(filename):
    with pytest.raises(DatasetError):
        parse_ravdess_filename(filename)


def test_incomplete_archive_includes_setup_instructions(tmp_path):
    actor_dir = tmp_path / "nested" / "Actor_01"
    actor_dir.mkdir(parents=True)
    (actor_dir / "03-01-01-01-01-01-01.wav").touch()
    (actor_dir / "03-02-01-01-01-01-01.wav").touch()
    with pytest.raises(DatasetError, match=r"Found 1 of 1440.*Audio_Speech_Actors_01-24\.zip"):
        inspect_dataset(tmp_path)


def test_duplicate_clip_rejected_across_nested_copies(tmp_path):
    for copy in ("first", "second"):
        actor_dir = tmp_path / copy / "Actor_01"
        actor_dir.mkdir(parents=True)
        (actor_dir / "03-01-01-01-01-01-01.wav").touch()
    with pytest.raises(DatasetError, match="Duplicate RAVDESS clip"):
        inspect_dataset(tmp_path)


def test_rejects_actor_folder_mismatch(tmp_path):
    actor_dir = tmp_path / "nested" / "Actor_01"
    actor_dir.mkdir(parents=True)
    (actor_dir / "03-01-01-01-01-01-02.wav").touch()
    with pytest.raises(DatasetError, match="disagrees with its Actor_01 folder"):
        inspect_dataset(tmp_path)


def test_actor_splits_are_disjoint_reproducible_and_balanced():
    frame = pd.DataFrame(
        [(actor, label) for actor in range(1, 25) for label in range(len(EMOTIONS))],
        columns=["actor", "label"],
    )
    split = split_by_actor(frame, seed=42)
    shuffled = split_by_actor(frame.sample(frac=1, random_state=4), seed=42)
    pd.testing.assert_frame_equal(split, shuffled.sort_index())
    assert split.groupby("actor")["split"].nunique().eq(1).all()
    assert split.groupby("split")["actor"].nunique().to_dict() == {
        "train": 16, "validation": 4, "test": 4,
    }
    for _, partition in split.groupby("split"):
        actors = partition["actor"].unique()
        assert (actors % 2).sum() * 2 == len(actors)
        assert set(partition["label"]) == set(range(len(EMOTIONS)))
