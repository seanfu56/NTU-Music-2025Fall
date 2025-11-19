"""Convert MIDI files into CP-word token sequences."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from miditoolkit import MidiFile


FEATURE_ORDER = ["tempo", "chord", "bar-beat", "type", "pitch", "duration", "velocity"]
BEAT_RESOLUTION = 480
BAR_RESOLUTION = BEAT_RESOLUTION * 4
STEP_RESOLUTION = BEAT_RESOLUTION // 4  # 16 positions per bar
MIDI_EXTENSIONS = {".mid", ".midi"}


def _extract_numeric_values(event_map: Dict, prefix: str) -> List[int]:
    values: List[int] = []
    for key in event_map.keys():
        if isinstance(key, str) and key.startswith(prefix):
            try:
                values.append(int(key.split("_")[-1]))
            except ValueError:
                continue
    return sorted(values)


def _nearest(value: int, candidates: Sequence[int]) -> int:
    if not candidates:
        raise ValueError("No candidates available for quantization.")
    return min(candidates, key=lambda cand: abs(cand - value))


def _quantize_tick(tick: int) -> int:
    return int(round(tick / STEP_RESOLUTION)) * STEP_RESOLUTION


class CPWordConverter:
    """Helper that turns MIDI files into CP-word id sequences."""

    def __init__(self, dictionary: Tuple[Dict, Dict]):
        self.event2word, self.word2event = dictionary
        self.tempo_bins = _extract_numeric_values(self.event2word["tempo"], "Tempo_")
        self.duration_bins = [
            v for v in _extract_numeric_values(self.event2word["duration"], "Note_Duration_") if v > 0
        ]
        self.velocity_bins = _extract_numeric_values(self.event2word["velocity"], "Note_Velocity_")
        pitch_bins = _extract_numeric_values(self.event2word["pitch"], "Note_Pitch_")
        self.min_pitch = min(pitch_bins) if pitch_bins else 21
        self.max_pitch = max(pitch_bins) if pitch_bins else 108
        self.available_chords = {
            key for key in self.event2word["chord"].keys() if isinstance(key, str) and key not in {"CONTI"}
        }

    def midi_to_words(self, midi_path: Path) -> np.ndarray:
        midi = MidiFile(str(midi_path))
        events = self._midi_to_events(midi)
        return self._events_to_ids(events)

    def _midi_to_events(self, midi: MidiFile) -> List[Dict]:
        notes = self._collect_notes(midi)
        if not notes:
            raise ValueError(f"No melody notes found in {midi.filename or 'MIDI file'}.")

        notes_by_tick: Dict[int, List[Tuple[int, int, int, int]]] = defaultdict(list)
        max_tick = 0
        for start, end, pitch, velocity in notes:
            notes_by_tick[start].append((start, end, pitch, velocity))
            if start > max_tick:
                max_tick = start

        tempo_events = sorted(
            [(int(round(ev.time)), float(ev.tempo)) for ev in midi.tempo_changes], key=lambda x: x[0]
        )
        if not tempo_events:
            tempo_events = [(0, 120.0)]
        max_tick = max(max_tick, tempo_events[-1][0])

        chord_events = sorted(
            [(int(round(marker.time)), marker.text.strip()) for marker in midi.markers], key=lambda x: x[0]
        )
        if chord_events:
            max_tick = max(max_tick, chord_events[-1][0])

        relevant_ticks = set(notes_by_tick.keys())
        relevant_ticks.update(_quantize_tick(t) for t, _ in tempo_events)
        relevant_ticks.update(_quantize_tick(t) for t, _ in chord_events)
        if not relevant_ticks:
            relevant_ticks.add(0)

        tempo_idx = 0
        current_tempo = tempo_events[0][1]
        chord_idx = -1
        current_chord = None
        last_tempo_token: str | None = None
        last_chord_token: str | None = None
        current_bar = -1

        events: List[Dict] = []
        for tick in sorted(relevant_ticks):
            bar_idx = tick // BAR_RESOLUTION
            while current_bar < bar_idx:
                current_bar += 1
                events.append(self._make_bar_event())

            tempo_changed = False
            while tempo_idx + 1 < len(tempo_events) and tempo_events[tempo_idx + 1][0] <= tick:
                tempo_idx += 1
                current_tempo = tempo_events[tempo_idx][1]
                tempo_changed = True
            tempo_token = self._tempo_token(current_tempo)
            if last_tempo_token is None:
                tempo_changed = True
            tempo_value = tempo_token if tempo_changed else "CONTI"
            if tempo_changed:
                last_tempo_token = tempo_token

            chord_changed = False
            while chord_idx + 1 < len(chord_events) and chord_events[chord_idx + 1][0] <= tick:
                chord_idx += 1
                current_chord = chord_events[chord_idx][1]
                chord_changed = True
            chord_token = None
            if current_chord and current_chord in self.available_chords:
                chord_token = current_chord
            chord_value = chord_token if (chord_changed and chord_token) else "CONTI"
            if chord_changed and chord_token:
                last_chord_token = chord_token
            elif last_chord_token is None and chord_token:
                chord_value = chord_token
                last_chord_token = chord_token

            beat_idx = (tick % BAR_RESOLUTION) // STEP_RESOLUTION
            events.append(
                {
                    "tempo": tempo_value,
                    "chord": chord_value if chord_value else "CONTI",
                    "bar-beat": f"Beat_{beat_idx}",
                    "type": "Metrical",
                    "pitch": 0,
                    "duration": 0,
                    "velocity": 0,
                }
            )

            for _, end, pitch, velocity in sorted(notes_by_tick.get(tick, []), key=lambda item: (item[2], item[3])):
                events.append(self._note_event(tick, end, pitch, velocity))

        # Append a trailing bar event to mimic the dataset layout before EOS.
        events.append(self._make_bar_event())
        events.append({"tempo": 0, "chord": 0, "bar-beat": 0, "type": "EOS", "pitch": 0, "duration": 0, "velocity": 0})
        return events

    def _collect_notes(self, midi: MidiFile) -> List[Tuple[int, int, int, int]]:
        notes: List[Tuple[int, int, int, int]] = []
        for inst in midi.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                start = _quantize_tick(note.start)
                end = _quantize_tick(note.end)
                if end <= start:
                    end = start + STEP_RESOLUTION
                notes.append((start, end, note.pitch, note.velocity))
        notes.sort(key=lambda item: (item[0], item[2], item[3]))
        return notes

    def _tempo_token(self, tempo_value: float) -> str:
        rounded = int(round(tempo_value))
        snapped = _nearest(rounded, self.tempo_bins)
        return f"Tempo_{snapped}"

    def _note_event(self, start: int, end: int, pitch: int, velocity: int) -> Dict:
        pitch_val = min(max(pitch, self.min_pitch), self.max_pitch)
        duration = max(end - start, STEP_RESOLUTION)
        duration_val = _nearest(duration, self.duration_bins)
        velocity_val = _nearest(velocity, self.velocity_bins)
        return {
            "tempo": 0,
            "chord": 0,
            "bar-beat": 0,
            "type": "Note",
            "pitch": f"Note_Pitch_{pitch_val}",
            "duration": f"Note_Duration_{duration_val}",
            "velocity": f"Note_Velocity_{velocity_val}",
        }

    @staticmethod
    def _make_bar_event() -> Dict:
        return {"tempo": 0, "chord": 0, "bar-beat": "Bar", "type": "Metrical", "pitch": 0, "duration": 0, "velocity": 0}

    def _events_to_ids(self, events: Sequence[Dict]) -> np.ndarray:
        rows: List[List[int]] = []
        for event in events:
            row: List[int] = []
            for key in FEATURE_ORDER:
                value = event[key]
                mapping = self.event2word[key]
                if value not in mapping:
                    if key in {"tempo", "chord"}:
                        value = "CONTI"
                    elif key == "bar-beat":
                        value = "Bar"
                    else:
                        value = 0
                row.append(mapping[value])
            rows.append(row)
        return np.asarray(rows, dtype=np.int16)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MIDI files to CP-word token sequences.")
    parser.add_argument("--input", "-i", required=True, help="Path to a MIDI file or a directory containing MIDI files.")
    parser.add_argument("--dictionary", "-d", required=True, help="Path to dictionary.pkl used for CPWord.")
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output file path or directory. When a directory is provided, .npy files will be created inside.",
    )
    parser.add_argument(
        "--format",
        choices=("npy", "json"),
        default="npy",
        help="Output format when writing into a directory (default: npy).",
    )
    return parser.parse_args()


def _collect_midi_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in MIDI_EXTENSIONS:
            raise ValueError(f"Unsupported file extension for {input_path}.")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"{input_path} does not exist.")

    midi_files: List[Path] = []
    for ext in MIDI_EXTENSIONS:
        midi_files.extend(sorted(input_path.rglob(f"*{ext}")))
        midi_files.extend(sorted(input_path.rglob(f"*{ext.upper()}")))
    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found under {input_path}.")
    return sorted(set(midi_files))


def _resolve_output_path(midi_path: Path, output_root: Path, as_directory: bool, fmt: str) -> Path:
    if as_directory:
        output_root.mkdir(parents=True, exist_ok=True)
        return output_root / f"{midi_path.stem}.{fmt}"
    if output_root.suffix.lower() in (".npy", ".json"):
        output_root.parent.mkdir(parents=True, exist_ok=True)
        return output_root
    output_root.parent.mkdir(parents=True, exist_ok=True)
    return output_root.with_suffix(f".{fmt}")


def _save_words(words: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(words.tolist(), f)
    else:
        np.save(output_path, words)


def main() -> None:
    args = _parse_args()
    with open(args.dictionary, "rb") as f:
        dictionary = pickle.load(f)
    converter = CPWordConverter(dictionary)

    input_path = Path(args.input)
    midi_files = _collect_midi_files(input_path)

    output_root = Path(args.output)
    treat_as_dir = input_path.is_dir() or output_root.is_dir() or len(midi_files) > 1
    fmt = args.format if args.format else "npy"

    for midi_file in midi_files:
        out_path = _resolve_output_path(midi_file, output_root, treat_as_dir, fmt)
        words = converter.midi_to_words(midi_file)
        _save_words(words, out_path)
        print(f"[mid2word] Saved {words.shape[0]} tokens for {midi_file} -> {out_path}")


if __name__ == "__main__":
    main()
