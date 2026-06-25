#!/usr/bin/env python3
"""Aggregate ROI quantification rows from multiple stack_analyzer pickle files."""

from __future__ import annotations

import argparse
import os
import pickle
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

from stack_analyzer import (
    BC_CORR_NORM_TRC_COLUMN,
    MARKED_EVENTS_COLUMN,
    ROI_QUANT_PICKLE_NAME,
    load_quant_store,
    parse_marked_events,
    parse_start_frames,
)

COLLECT_PICKLE_NAME = "pickle_stack_collect.pkl"

COLLECT_COLUMNS = [
    "directory",
    "starts",
    "freq + avr",
    "extension",
    "BC-corr. Norm-Trc",
    "Marked events",
    "ROI #",
    "Evoked Events",
    "Marked Events",
    "Non-events",
]


def collect_pickle_path_for_directory(directory: str | Path) -> Path:
    return Path(directory).resolve() / COLLECT_PICKLE_NAME


def empty_collect_store(collect_directory: str | Path) -> dict:
    return {
        "version": 1,
        "columns": list(COLLECT_COLUMNS),
        "collect_directory": str(Path(collect_directory).resolve()),
        "experiments": [],
        "rows": [],
    }


def load_collect_store(path: Path) -> dict:
    if not path.exists():
        return empty_collect_store(path.parent)
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict) or "rows" not in data:
        return empty_collect_store(path.parent)
    data["columns"] = list(COLLECT_COLUMNS)
    data.setdefault("experiments", [])
    data.setdefault("collect_directory", str(path.parent.resolve()))
    for row in data.get("rows", []):
        for column in COLLECT_COLUMNS:
            if column not in row:
                row[column] = None
    return data


def save_collect_store(path: Path, store: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(store, handle, protocol=pickle.HIGHEST_PROTOCOL)


def resolve_experiment_pickle(path_str: str) -> Path | None:
    path = Path(path_str).resolve()
    if path.is_file() and path.suffix.lower() == ".pkl":
        return path
    if path.is_dir():
        candidate = path / ROI_QUANT_PICKLE_NAME
        if candidate.exists():
            return candidate
    return None


def merge_intervals_inclusive(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    normalized = [(min(a, b), max(a, b)) for a, b in intervals]
    normalized.sort(key=lambda pair: pair[0])
    merged = [normalized[0]]
    for start, end in normalized[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def complement_intervals_inclusive(
    occupied: list[tuple[int, int]], n_frames: int
) -> list[tuple[int, int]]:
    if n_frames <= 0:
        return []
    merged = merge_intervals_inclusive(occupied)
    gaps: list[tuple[int, int]] = []
    cursor = 0
    for start, end in merged:
        if start > cursor:
            gaps.append((cursor, start - 1))
        cursor = max(cursor, end + 1)
    if cursor < n_frames:
        gaps.append((cursor, n_frames - 1))
    return [(start, end) for start, end in gaps if end >= start]


def evoked_intervals_exclusive(row: dict, n_frames: int) -> list[tuple[int, int]]:
    try:
        extension = int(float(row.get("extension", "50")))
    except (TypeError, ValueError):
        extension = 50
    starts = parse_start_frames(str(row.get("starts", "")), n_frames)
    intervals: list[tuple[int, int]] = []
    for start in starts:
        end_exclusive = min(start + extension, n_frames)
        if start < end_exclusive:
            intervals.append((start, end_exclusive))
    return intervals


def evoked_intervals_inclusive(row: dict, n_frames: int) -> list[tuple[int, int]]:
    return [(start, end - 1) for start, end in evoked_intervals_exclusive(row, n_frames)]


def extract_exclusive_segments(
    trace: np.ndarray, intervals: list[tuple[int, int]]
) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    n_frames = len(trace)
    for start, end_exclusive in intervals:
        start = max(0, min(int(start), n_frames - 1))
        end_exclusive = max(start, min(int(end_exclusive), n_frames))
        segment = trace[start:end_exclusive]
        if segment.size:
            segments.append(np.asarray(segment, dtype=np.float64))
    return segments


def extract_inclusive_segments(
    trace: np.ndarray, intervals: list[tuple[int, int]]
) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    n_frames = len(trace)
    for start, end in intervals:
        start = max(0, min(int(start), n_frames - 1))
        end = max(start, min(int(end), n_frames - 1))
        segment = trace[start : end + 1]
        if segment.size:
            segments.append(np.asarray(segment, dtype=np.float64))
    return segments


def build_collect_row(source_row: dict, roi_number: int) -> dict | None:
    norm_trace = source_row.get(BC_CORR_NORM_TRC_COLUMN)
    if norm_trace is None:
        return None
    trace = np.asarray(norm_trace, dtype=np.float64)
    if trace.size == 0:
        return None

    n_frames = len(trace)
    evoked_exclusive = evoked_intervals_exclusive(source_row, n_frames)
    marked_inclusive = parse_marked_events(source_row.get(MARKED_EVENTS_COLUMN))
    occupied = merge_intervals_inclusive(
        evoked_intervals_inclusive(source_row, n_frames) + marked_inclusive
    )
    non_event_inclusive = complement_intervals_inclusive(occupied, n_frames)

    return {
        "directory": source_row.get("directory"),
        "starts": source_row.get("starts"),
        "freq + avr": source_row.get("freq + avr"),
        "extension": source_row.get("extension"),
        "BC-corr. Norm-Trc": trace.copy(),
        "Marked events": source_row.get(MARKED_EVENTS_COLUMN),
        "ROI #": int(roi_number),
        "Evoked Events": extract_exclusive_segments(trace, evoked_exclusive),
        "Marked Events": extract_inclusive_segments(trace, marked_inclusive),
        "Non-events": extract_inclusive_segments(trace, non_event_inclusive),
    }


def rebuild_collect_rows(experiment_pickle_paths: list[str]) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    warnings: list[str] = []

    for pickle_path_str in experiment_pickle_paths:
        pickle_path = Path(pickle_path_str).resolve()
        if not pickle_path.exists():
            warnings.append(f"Missing pickle: {pickle_path}")
            continue

        store = load_quant_store(pickle_path)
        for roi_index, source_row in enumerate(store.get("rows", [])):
            collect_row = build_collect_row(source_row, roi_number=roi_index + 1)
            if collect_row is None:
                directory = source_row.get("directory", pickle_path.parent)
                warnings.append(
                    f"Skipped ROI {roi_index + 1} in {pickle_path.name} "
                    f"({directory}): no {BC_CORR_NORM_TRC_COLUMN!r}."
                )
                continue
            rows.append(collect_row)

    return rows, warnings


class StackAnalyzerTotalApp:
    def __init__(self, initial_dir: str | None = None) -> None:
        self.collect_directory = (
            str(Path(initial_dir).resolve()) if initial_dir else str(Path.cwd().resolve())
        )
        self.store = empty_collect_store(self.collect_directory)
        self.collect_path = collect_pickle_path_for_directory(self.collect_directory)

        self.root = tk.Tk()
        self.root.title("Stack Analyzer Total")
        self.root.geometry("920x520")
        self.root.minsize(720, 420)

        self._build_ui()
        self._load_collect_directory(self.collect_directory, prompt_create=False)

    def _build_ui(self) -> None:
        header = tk.Frame(self.root, padx=10, pady=8)
        header.pack(fill="x")

        tk.Label(header, text="Collect directory:").pack(side="left")
        self.dir_var = tk.StringVar(value=self.collect_directory)
        self.dir_entry = tk.Entry(header, textvariable=self.dir_var, width=72)
        self.dir_entry.pack(side="left", padx=6)
        tk.Button(header, text="Browse…", command=self._browse_collect_directory).pack(side="left")
        tk.Button(header, text="Load", command=self._load_from_entry).pack(side="left", padx=(6, 0))

        info = tk.Frame(self.root, padx=10)
        info.pack(fill="x", pady=(0, 8))
        self.collect_file_label = tk.Label(info, anchor="w", justify="left")
        self.collect_file_label.pack(fill="x")
        self.summary_label = tk.Label(info, anchor="w", justify="left")
        self.summary_label.pack(fill="x")

        body = tk.Frame(self.root, padx=10, pady=4)
        body.pack(fill="both", expand=True)

        list_frame = tk.Frame(body)
        list_frame.pack(side="left", fill="both", expand=True)

        tk.Label(list_frame, text="Experiments (source ROI quantification pickles)").pack(
            anchor="w"
        )
        list_container = tk.Frame(list_frame)
        list_container.pack(fill="both", expand=True, pady=(4, 0))

        scrollbar = ttk.Scrollbar(list_container, orient="vertical")
        self.experiment_listbox = tk.Listbox(
            list_container,
            height=16,
            yscrollcommand=scrollbar.set,
            selectmode=tk.SINGLE,
        )
        scrollbar.config(command=self.experiment_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.experiment_listbox.pack(side="left", fill="both", expand=True)

        button_col = tk.Frame(body)
        button_col.pack(side="right", fill="y", padx=(10, 0))
        tk.Button(button_col, text="Add Experiment", width=16, command=self._add_experiment).pack(
            pady=(0, 6)
        )
        tk.Button(
            button_col, text="Remove Experiment", width=16, command=self._remove_experiment
        ).pack(pady=(0, 6))
        tk.Button(button_col, text="Rebuild Collection", width=16, command=self._rebuild_and_save).pack(
            pady=(0, 6)
        )

        footer = tk.Frame(self.root, padx=10, pady=8)
        footer.pack(fill="both", expand=False)
        tk.Label(footer, text="Status").pack(anchor="w")
        self.status_text = tk.Text(footer, height=6, wrap="word")
        self.status_text.pack(fill="both", expand=True, pady=(4, 0))
        self.status_text.config(state="disabled")

    def _set_status(self, message: str) -> None:
        self.status_text.config(state="normal")
        self.status_text.delete("1.0", "end")
        self.status_text.insert("1.0", message)
        self.status_text.config(state="disabled")

    def _refresh_experiment_list(self) -> None:
        self.experiment_listbox.delete(0, "end")
        for path_str in self.store.get("experiments", []):
            self.experiment_listbox.insert("end", path_str)

    def _refresh_summary(self) -> None:
        self.collect_file_label.config(
            text=f"Collection file: {self.collect_path}"
            + ("  (exists)" if self.collect_path.exists() else "  (will be created on save)")
        )
        self.summary_label.config(
            text=(
                f"Experiments: {len(self.store.get('experiments', []))}  |  "
                f"Collected rows: {len(self.store.get('rows', []))}"
            )
        )

    def _browse_collect_directory(self) -> None:
        path = filedialog.askdirectory(
            title="Select collection directory",
            initialdir=self.collect_directory,
        )
        if path:
            self._load_collect_directory(path)

    def _load_from_entry(self) -> None:
        path = self.dir_var.get().strip()
        if not path:
            messagebox.showinfo("Load", "Enter or browse to a collection directory.")
            return
        self._load_collect_directory(path)

    def _load_collect_directory(self, directory: str, *, prompt_create: bool = True) -> None:
        resolved = str(Path(directory).resolve())
        self.collect_directory = resolved
        self.dir_var.set(resolved)
        self.collect_path = collect_pickle_path_for_directory(resolved)

        if self.collect_path.exists():
            self.store = load_collect_store(self.collect_path)
            self.store["collect_directory"] = resolved
        else:
            if prompt_create and not messagebox.askyesno(
                "Create collection file",
                f"No {COLLECT_PICKLE_NAME} found in:\n{resolved}\n\nCreate a new collection file?",
            ):
                self.store = empty_collect_store(resolved)
            else:
                self.store = empty_collect_store(resolved)

        self._refresh_experiment_list()
        self._refresh_summary()
        self._set_status(f"Loaded collection directory:\n{resolved}")

    def _save(self) -> None:
        self.store["collect_directory"] = self.collect_directory
        save_collect_store(self.collect_path, self.store)
        self._refresh_summary()
        self._set_status(f"Saved {self.collect_path}\nRows: {len(self.store.get('rows', []))}")

    def _rebuild_and_save(self) -> None:
        experiments = list(self.store.get("experiments", []))
        rows, warnings = rebuild_collect_rows(experiments)
        self.store["rows"] = rows

        lines = [f"Rebuilt collection with {len(rows)} row(s)."]
        if warnings:
            lines.append("")
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in warnings[:20])
            if len(warnings) > 20:
                lines.append(f"- … and {len(warnings) - 20} more")
        self._save()
        self._set_status("\n".join(lines))

    def _add_experiment(self) -> None:
        path = filedialog.askopenfilename(
            title="Select experiment ROI quantification pickle",
            initialdir=self.collect_directory,
            filetypes=[
                ("Pickle files", "*.pkl"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            directory = filedialog.askdirectory(
                title="Or select experiment folder containing ROI quantification pickle",
                initialdir=self.collect_directory,
            )
            if not directory:
                return
            path = str(Path(directory) / ROI_QUANT_PICKLE_NAME)

        pickle_path = resolve_experiment_pickle(path)
        if pickle_path is None:
            messagebox.showerror(
                "Add Experiment",
                f"Could not find a valid pickle file at:\n{path}",
            )
            return

        pickle_str = str(pickle_path.resolve())
        experiments = list(self.store.get("experiments", []))
        if pickle_str in experiments:
            messagebox.showinfo("Add Experiment", "That experiment is already in the collection.")
            return

        experiments.append(pickle_str)
        self.store["experiments"] = experiments
        self._refresh_experiment_list()
        self._rebuild_and_save()

    def _remove_experiment(self) -> None:
        selection = self.experiment_listbox.curselection()
        if not selection:
            messagebox.showinfo("Remove Experiment", "Select an experiment to remove.")
            return

        index = int(selection[0])
        experiments = list(self.store.get("experiments", []))
        if index < 0 or index >= len(experiments):
            return

        removed = experiments.pop(index)
        if not messagebox.askyesno(
            "Remove Experiment",
            f"Remove experiment and its collected rows?\n\n{removed}",
        ):
            return

        self.store["experiments"] = experiments
        self._refresh_experiment_list()
        self._rebuild_and_save()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate stack_analyzer ROI quantification pickles"
    )
    parser.add_argument(
        "--collect-dir",
        help=f"Directory containing (or to create) {COLLECT_PICKLE_NAME}",
    )
    args = parser.parse_args()
    app = StackAnalyzerTotalApp(initial_dir=args.collect_dir)
    app.run()


if __name__ == "__main__":
    main()
