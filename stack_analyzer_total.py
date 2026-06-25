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

from portable_paths import path_for_storage, path_from_storage, resolve_directory
from stack_analyzer import (
    BC_CORR_NORM_TRC_COLUMN,
    MARKED_EVENTS_COLUMN,
    ROI_QUANT_PICKLE_NAME,
    format_quant_value_detail,
    load_quant_store,
    parse_marked_events,
    parse_start_frames,
    summarize_quant_cell,
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


def normalize_collect_store_paths(
    store: dict,
    collect_directory: str | Path,
    *,
    stored_base: str | Path | None = None,
) -> None:
    """Rewrite experiment and row directory paths as collect-directory-relative."""
    base = Path(collect_directory).resolve()
    anchor = stored_base if stored_base is not None else store.get("collect_directory")

    store["experiments"] = [
        path_for_storage(
            path_from_storage(path_str, base, stored_base=anchor),
            base,
        )
        for path_str in store.get("experiments", [])
    ]
    for row in store.get("rows", []):
        directory = row.get("directory")
        if directory:
            row["directory"] = path_for_storage(
                path_from_storage(str(directory), base, stored_base=anchor),
                base,
            )
    store["collect_directory"] = str(base)


def resolve_collect_directory_value(
    directory,
    collect_directory: str | Path,
    *,
    stored_base: str | Path | None = None,
) -> str | None:
    if directory is None:
        return None
    text = str(directory).strip()
    if not text:
        return None
    if stored_base is not None:
        return str(path_from_storage(text, collect_directory, stored_base=stored_base))
    return resolve_directory(text, collect_directory)


def iter_stored_path_strings(store: dict) -> list[str]:
    paths = list(store.get("experiments", []))
    for row in store.get("rows", []):
        directory = row.get("directory")
        if directory:
            paths.append(str(directory))
    return paths


def stale_absolute_paths_in_store(store: dict) -> list[Path]:
    stale: list[Path] = []
    for path_str in iter_stored_path_strings(store):
        path = Path(path_str)
        if path.is_absolute() and not path.resolve().exists():
            stale.append(path.resolve())
    return stale


def _same_drive(path_a: str | Path, path_b: str | Path) -> bool:
    drive_a = os.path.splitdrive(str(Path(path_a)))[0].lower()
    drive_b = os.path.splitdrive(str(Path(path_b)))[0].lower()
    return bool(drive_a) and drive_a == drive_b


def infer_path_anchor_from_stale_paths(
    stale_paths: list[Path],
    current_collect_directory: str | Path,
) -> Path | None:
    if not stale_paths:
        return None

    parents: list[Path] = []
    for path in stale_paths:
        if path.suffix.lower() == ".pkl":
            parents.append(path.parent)
        else:
            parents.append(path)

    try:
        common = Path(os.path.commonpath([str(parent) for parent in parents]))
    except ValueError:
        return None

    collect_name = Path(current_collect_directory).name
    if common.name == collect_name:
        return common.resolve()
    return (common.parent / collect_name).resolve()


def resolve_path_anchor(store: dict, current_collect_directory: str | Path) -> Path:
    """Pick the best collect-directory anchor for remapping stored paths."""
    current = Path(current_collect_directory).resolve()
    stale_paths = stale_absolute_paths_in_store(store)
    if not stale_paths:
        declared = store.get("path_anchor") or store.get("collect_directory")
        return Path(declared).resolve() if declared else current

    for anchor_key in ("path_anchor", "collect_directory"):
        declared = store.get(anchor_key)
        if not declared:
            continue
        declared_path = Path(declared).resolve()
        if _same_drive(declared_path, stale_paths[0]):
            return declared_path

    inferred = infer_path_anchor_from_stale_paths(stale_paths, current)
    if inferred is not None:
        return inferred
    return current


def effective_stored_base(
    store: dict,
    current_collect_directory: str | Path,
    *,
    explicit_stored_base: str | Path | None = None,
) -> Path:
    if explicit_stored_base is not None:
        return Path(explicit_stored_base).resolve()
    return resolve_path_anchor(store, current_collect_directory)

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


def build_collect_row(
    source_row: dict,
    roi_number: int,
    *,
    collect_directory: str | Path,
    stored_collect_directory: str | Path | None = None,
    experiment_pickle_path: Path | None = None,
) -> dict | None:
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

    row_directory = source_row.get("directory")
    if row_directory:
        resolved_directory = path_from_storage(
            str(row_directory),
            collect_directory,
            stored_base=stored_collect_directory,
        )
        if not resolved_directory.exists() and experiment_pickle_path is not None:
            resolved_directory = experiment_pickle_path.parent.resolve()
    elif experiment_pickle_path is not None:
        resolved_directory = experiment_pickle_path.parent.resolve()
    else:
        resolved_directory = None

    return {
        "directory": str(resolved_directory) if resolved_directory is not None else None,
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


def rebuild_collect_rows(
    experiment_pickle_paths: list[str],
    *,
    collect_directory: str | Path,
    stored_collect_directory: str | Path | None = None,
) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    warnings: list[str] = []

    for stored_path in experiment_pickle_paths:
        pickle_path = path_from_storage(
            stored_path,
            collect_directory,
            stored_base=stored_collect_directory,
        )
        if not pickle_path.exists():
            warnings.append(f"Missing pickle: {pickle_path}")
            continue

        store = load_quant_store(pickle_path)
        for roi_index, source_row in enumerate(store.get("rows", [])):
            collect_row = build_collect_row(
                source_row,
                roi_number=roi_index + 1,
                collect_directory=collect_directory,
                stored_collect_directory=stored_collect_directory,
                experiment_pickle_path=pickle_path,
            )
            if collect_row is None:
                directory = source_row.get("directory", pickle_path.parent)
                warnings.append(
                    f"Skipped ROI {roi_index + 1} in {pickle_path.name} "
                    f"({directory}): no {BC_CORR_NORM_TRC_COLUMN!r}."
                )
                continue
            rows.append(collect_row)

    return rows, warnings


SEGMENT_LIST_COLUMNS = frozenset({"Evoked Events", "Marked Events", "Non-events"})


def summarize_collect_cell(
    column: str,
    value,
    *,
    collect_directory: str | Path | None = None,
    stored_base: str | Path | None = None,
) -> str:
    if column == "directory" and collect_directory is not None:
        value = resolve_collect_directory_value(
            value, collect_directory, stored_base=stored_base
        )
    if column in SEGMENT_LIST_COLUMNS:
        if not value:
            return ""
        if not isinstance(value, list):
            return str(value)
        total_pts = sum(
            len(segment) for segment in value if isinstance(segment, np.ndarray)
        )
        return f"{len(value)} segment(s), {total_pts} pts"
    if column == "ROI #":
        return "" if value is None else str(value)
    return summarize_quant_cell(column, value)


def format_segment_list_detail(segments) -> str:
    if not segments:
        return "[]"
    if not isinstance(segments, list):
        return str(segments)
    lines: list[str] = [f"{len(segments)} segment(s)"]
    for index, segment in enumerate(segments):
        if not isinstance(segment, np.ndarray) or segment.size == 0:
            lines.append(f"\n[{index}] (empty)")
            continue
        lines.append(
            f"\n[{index}] length={segment.size}, "
            f"min={float(np.nanmin(segment)):.6g}, "
            f"max={float(np.nanmax(segment)):.6g}, "
            f"mean={float(np.nanmean(segment)):.6g}"
        )
        if segment.size <= 16:
            lines.append(
                np.array2string(segment, precision=4, separator=", ", max_line_width=100)
            )
        else:
            head = np.array2string(segment[:8], precision=4, separator=", ")
            tail = np.array2string(segment[-4:], precision=4, separator=", ")
            lines.append(f"{head} … ({segment.size} total) … {tail}")
    return "\n".join(lines)


def format_collect_value_detail(
    column: str,
    value,
    *,
    collect_directory: str | Path | None = None,
    stored_base: str | Path | None = None,
) -> str:
    if column == "directory" and collect_directory is not None:
        value = resolve_collect_directory_value(
            value, collect_directory, stored_base=stored_base
        )
    if column in SEGMENT_LIST_COLUMNS:
        return format_segment_list_detail(value)
    return format_quant_value_detail(column, value)


def format_collect_row_detail(
    row: dict,
    row_index: int,
    columns: list[str],
    *,
    collect_directory: str | Path | None = None,
    stored_base: str | Path | None = None,
) -> str:
    lines = [f"Row {row_index}"]
    for column in columns:
        if column not in row:
            continue
        lines.append(f"\n{column}:")
        lines.append(
            format_collect_value_detail(
                column,
                row[column],
                collect_directory=collect_directory,
                stored_base=stored_base,
            )
        )
    return "\n".join(lines)


def open_collect_pickle_inspector(
    path: Path,
    *,
    parent: tk.Misc | None = None,
    collect_directory: str | Path | None = None,
    stored_base: str | Path | None = None,
) -> None:
    """Open a scrollable overview of the collection pickle file."""
    store = load_collect_store(path)
    columns = list(store.get("columns", COLLECT_COLUMNS))
    rows = store.get("rows", [])
    experiments = store.get("experiments", [])
    current_base = (
        Path(collect_directory).resolve()
        if collect_directory is not None
        else path.parent.resolve()
    )
    resolved_stored_base = effective_stored_base(
        store,
        current_base,
        explicit_stored_base=stored_base,
    )
    tree_columns = ["#"] + columns

    owns_root = parent is None
    if owns_root:
        parent = tk.Tk()
        parent.withdraw()

    window = tk.Toplevel(parent)
    window.title(f"Inspect Pickle — {path.name}")
    window.geometry("1100x700")
    window.minsize(640, 420)

    header = tk.Frame(window, padx=10, pady=8)
    header.pack(fill="x")
    tk.Label(
        header,
        text=f"File: {path}",
        anchor="w",
        justify="left",
        wraplength=1040,
    ).pack(fill="x")
    tk.Label(
        header,
        text=(
            f"version={store.get('version', '?')}  |  "
            f"{len(experiments)} experiment(s)  |  {len(rows)} row(s)"
        ),
        anchor="w",
    ).pack(fill="x")
    if experiments:
        experiment_lines = [
            str(
                path_from_storage(
                    experiment_path,
                    current_base,
                    stored_base=resolved_stored_base,
                )
            )
            for experiment_path in experiments
        ]
        tk.Label(
            header,
            text="Experiments:\n" + "\n".join(experiment_lines),
            anchor="w",
            justify="left",
            wraplength=1040,
        ).pack(fill="x", pady=(4, 0))

    paned = tk.PanedWindow(window, orient="vertical", sashrelief="raised", sashwidth=4)
    paned.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    table_frame = tk.Frame(paned)
    paned.add(table_frame, minsize=180)

    table_scroll_y = ttk.Scrollbar(table_frame, orient="vertical")
    table_scroll_x = ttk.Scrollbar(table_frame, orient="horizontal")
    tree = ttk.Treeview(
        table_frame,
        columns=tree_columns,
        show="headings",
        yscrollcommand=table_scroll_y.set,
        xscrollcommand=table_scroll_x.set,
        selectmode="browse",
    )
    table_scroll_y.config(command=tree.yview)
    table_scroll_x.config(command=tree.xview)
    table_scroll_y.pack(side="right", fill="y")
    table_scroll_x.pack(side="bottom", fill="x")
    tree.pack(side="left", fill="both", expand=True)

    default_widths = {
        "#": 40,
        "directory": 220,
        "starts": 150,
        "freq + avr": 110,
        "extension": 70,
        "BC-corr. Norm-Trc": 130,
        "Marked events": 120,
        "ROI #": 50,
        "Evoked Events": 120,
        "Marked Events": 120,
        "Non-events": 120,
    }
    for column in tree_columns:
        heading = "#" if column == "#" else column
        tree.heading(column, text=heading)
        tree.column(column, width=default_widths.get(column, 100), minwidth=48, stretch=False)

    detail_frame = tk.Frame(paned)
    paned.add(detail_frame, minsize=140)
    tk.Label(detail_frame, text="Row details", anchor="w").pack(fill="x", pady=(0, 4))
    detail_outer = tk.Frame(detail_frame)
    detail_outer.pack(fill="both", expand=True)
    detail_scroll_y = ttk.Scrollbar(detail_outer, orient="vertical")
    detail_scroll_x = ttk.Scrollbar(detail_outer, orient="horizontal")
    detail_text = tk.Text(
        detail_outer,
        wrap="none",
        font=("Consolas", 10),
        yscrollcommand=detail_scroll_y.set,
        xscrollcommand=detail_scroll_x.set,
    )
    detail_scroll_y.config(command=detail_text.yview)
    detail_scroll_x.config(command=detail_text.xview)
    detail_scroll_y.pack(side="right", fill="y")
    detail_scroll_x.pack(side="bottom", fill="x")
    detail_text.pack(side="left", fill="both", expand=True)
    detail_text.config(state="disabled")

    for row_index, row in enumerate(rows):
        values = [str(row_index)]
        for column in columns:
            values.append(
                summarize_collect_cell(
                    column,
                    row.get(column),
                    collect_directory=current_base,
                    stored_base=resolved_stored_base,
                )
            )
        tree.insert("", "end", iid=str(row_index), values=values)

    def show_row_detail(_event=None) -> None:
        selection = tree.selection()
        detail_text.config(state="normal")
        detail_text.delete("1.0", "end")
        if not selection:
            detail_text.config(state="disabled")
            return
        row_index = int(selection[0])
        if 0 <= row_index < len(rows):
            detail_text.insert(
                "1.0",
                format_collect_row_detail(
                    rows[row_index],
                    row_index,
                    columns,
                    collect_directory=current_base,
                    stored_base=resolved_stored_base,
                ),
            )
        detail_text.config(state="disabled")

    tree.bind("<<TreeviewSelect>>", show_row_detail)
    if rows:
        tree.selection_set("0")
        tree.focus("0")
        show_row_detail()

    def on_close() -> None:
        window.destroy()
        if owns_root:
            parent.destroy()

    window.protocol("WM_DELETE_WINDOW", on_close)


class StackAnalyzerTotalApp:
    def __init__(self, initial_dir: str | None = None) -> None:
        self.collect_directory = (
            str(Path(initial_dir).resolve()) if initial_dir else str(Path.cwd().resolve())
        )
        self.stored_collect_directory = self.collect_directory
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
        tk.Button(button_col, text="Inspect Pickle", width=16, command=self._on_inspect_pickle).pack(
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
            resolved = path_from_storage(
                path_str,
                self.collect_directory,
                stored_base=self.stored_collect_directory,
            )
            self.experiment_listbox.insert("end", str(resolved))

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
            self.stored_collect_directory = (
                self.store.get("path_anchor")
                or self.store.get("collect_directory", resolved)
            )
        else:
            if prompt_create and not messagebox.askyesno(
                "Create collection file",
                f"No {COLLECT_PICKLE_NAME} found in:\n{resolved}\n\nCreate a new collection file?",
            ):
                self.store = empty_collect_store(resolved)
            else:
                self.store = empty_collect_store(resolved)
            self.stored_collect_directory = resolved

        self.collect_directory = resolved
        self.store["collect_directory"] = resolved

        self._refresh_experiment_list()
        self._refresh_summary()
        self._set_status(f"Loaded collection directory:\n{resolved}")

    def _save(self) -> None:
        self.store["collect_directory"] = self.collect_directory
        self.store["path_anchor"] = self.stored_collect_directory
        normalize_collect_store_paths(
            self.store,
            self.collect_directory,
            stored_base=self.stored_collect_directory,
        )
        save_collect_store(self.collect_path, self.store)
        self.stored_collect_directory = self.collect_directory
        self._refresh_experiment_list()
        self._refresh_summary()
        self._set_status(f"Saved {self.collect_path}\nRows: {len(self.store.get('rows', []))}")

    def _rebuild_and_save(self) -> None:
        experiments = list(self.store.get("experiments", []))
        rows, warnings = rebuild_collect_rows(
            experiments,
            collect_directory=self.collect_directory,
            stored_collect_directory=self.stored_collect_directory,
        )
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

        experiments = list(self.store.get("experiments", []))
        resolved_new = pickle_path.resolve()
        for existing in experiments:
            resolved_existing = path_from_storage(
                existing,
                self.collect_directory,
                stored_base=self.stored_collect_directory,
            )
            if resolved_existing.resolve() == resolved_new:
                messagebox.showinfo("Add Experiment", "That experiment is already in the collection.")
                return

        experiments.append(path_for_storage(resolved_new, self.collect_directory))
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
        removed_display = str(
            path_from_storage(
                removed,
                self.collect_directory,
                stored_base=self.stored_collect_directory,
            )
        )
        if not messagebox.askyesno(
            "Remove Experiment",
            f"Remove experiment and its collected rows?\n\n{removed_display}",
        ):
            experiments.insert(index, removed)
            return

        self.store["experiments"] = experiments
        self._refresh_experiment_list()
        self._rebuild_and_save()

    def _on_inspect_pickle(self) -> None:
        path = self.collect_path
        if not path.exists():
            path_str = filedialog.askopenfilename(
                title="Select collection pickle",
                initialdir=self.collect_directory,
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            )
            if not path_str:
                return
            path = Path(path_str)
        open_collect_pickle_inspector(
            path,
            parent=self.root,
            collect_directory=self.collect_directory,
            stored_base=self.stored_collect_directory,
        )

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
