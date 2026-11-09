"""Discovery, validation, and actor-level splitting for RAVDESS speech."""

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import re

import numpy as np
import pandas as pd
import soundfile as sf

from config import DATASET_URL, EMOTIONS


DATASET_SETUP = (
    f"Download Audio_Speech_Actors_01-24.zip from {DATASET_URL} "
    "and extract it under dataset/ or pass its extracted directory with --dataset-dir."
)
ACTORS = tuple(range(1, 25))
SPLITS = ("train", "validation", "test")
ACTOR_DIRECTORY = re.compile(r"Actor_([0-9]{2})", re.IGNORECASE)
FILE_STEM = re.compile(r"[0-9]{2}(?:-[0-9]{2}){6}")


class DatasetError(ValueError):
    """The supplied data cannot support the RAVDESS experiment."""


@dataclass(frozen=True)
class RavdessClip:
    actor: int
    label: int
    intensity: int
    statement: int
    repetition: int

    @property
    def emotion(self) -> str:
        return EMOTIONS[self.label]

    @property
    def filename(self) -> str:
        return (
            f"03-01-{self.label + 1:02d}-{self.intensity:02d}-"
            f"{self.statement:02d}-{self.repetition:02d}-{self.actor:02d}.wav"
        )


def parse_ravdess_filename(filename: str | Path) -> RavdessClip:
    """Parse a canonical audio-only speech filename into its experiment metadata."""
    path = Path(filename)
    if path.suffix.lower() != ".wav" or not FILE_STEM.fullmatch(path.stem):
        raise DatasetError(f"Invalid RAVDESS WAV filename {path.name!r}.")

    modality, channel, emotion, intensity, statement, repetition, actor = map(
        int, path.stem.split("-")
    )
    if modality != 3 or channel != 1:
        raise DatasetError(f"Expected an audio-only speech file, found {path.name!r}.")
    if (
        emotion not in range(1, len(EMOTIONS) + 1)
        or actor not in ACTORS
        or intensity not in (1, 2)
        or (emotion == 1 and intensity != 1)
        or statement not in (1, 2)
        or repetition not in (1, 2)
    ):
        raise DatasetError(f"Invalid RAVDESS metadata in {path.name!r}.")
    return RavdessClip(actor, emotion - 1, intensity, statement, repetition)


def _actor_folder(path: Path, root: Path) -> int | None:
    for directory in path.parents:
        match = ACTOR_DIRECTORY.fullmatch(directory.name)
        if match:
            return int(match[1])
        if directory == root:
            break
    return None


def inspect_dataset(root: str | Path) -> pd.DataFrame:
    """Validate the complete speech archive and return a deterministic clip inventory."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise DatasetError(f"Dataset directory {root} was not found. {DATASET_SETUP}")

    inventory = {}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() != ".wav" or not path.is_file():
            continue
        actor = _actor_folder(path, root)
        if actor is None or path.stem.startswith("03-02-"):
            continue
        clip = parse_ravdess_filename(path)
        if clip.actor != actor:
            raise DatasetError(f"Actor ID in {path.name} disagrees with its Actor_{actor:02d} folder.")
        if clip in inventory:
            raise DatasetError(f"Duplicate RAVDESS clip {clip.filename} at {inventory[clip]} and {path}.")
        inventory[clip] = path

    expected = {
        RavdessClip(actor, label, intensity, statement, repetition)
        for actor, label, statement, repetition in product(ACTORS, range(len(EMOTIONS)), (1, 2), (1, 2))
        for intensity in ((1,) if label == 0 else (1, 2))
    }
    if missing := expected.difference(inventory):
        example = min(clip.filename for clip in missing)
        raise DatasetError(
            f"Incomplete RAVDESS speech archive in {root}. Found {len(inventory)} of "
            f"{len(expected)} expected clips. Missing example: {example}. {DATASET_SETUP}"
        )

    records = []
    for clip, path in inventory.items():
        try:
            info = sf.info(str(path))
        except (OSError, RuntimeError, ValueError) as exc:
            raise DatasetError(f"Could not read {path}. Re-extract the WAV from the speech archive.") from exc
        if info.format not in {"WAV", "WAVEX", "RF64"} or info.frames <= 0 or info.samplerate <= 0:
            raise DatasetError(f"Invalid or empty WAV file {path}. Re-extract it from the speech archive.")
        records.append({
            "path": path,
            "actor": clip.actor,
            "emotion": clip.emotion,
            "label": clip.label,
            "duration_seconds": info.duration,
        })
    return pd.DataFrame.from_records(records)


def split_by_actor(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Assign 16 training, four validation, and four test actors using a fixed seed."""
    if set(frame["actor"]) != set(ACTORS):
        raise DatasetError("Actor splitting requires the complete set of 24 RAVDESS speech actors.")

    rng = np.random.default_rng(seed)
    assignment = {}
    # The dataset documents odd/even actor IDs as male/female, respectively.
    for group in (ACTORS[::2], ACTORS[1::2]):
        train, validation, test = np.split(rng.permutation(group), (8, 10))
        for split, actors in zip(SPLITS, (train, validation, test), strict=True):
            assignment.update(dict.fromkeys(actors, split))

    result = frame.assign(split=frame["actor"].map(assignment))
    labels = set(range(len(EMOTIONS)))
    for split in SPLITS:
        if set(result.loc[result["split"] == split, "label"]) != labels:
            raise DatasetError(f"The {split} split must contain all eight emotion labels.")
    return result
