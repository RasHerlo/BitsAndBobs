#!/usr/bin/env python3
"""Aggregate ROI quantification rows from multiple stack_analyzer pickle files."""

from __future__ import annotations

import argparse
import os
import pickle
import textwrap
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from portable_paths import path_for_storage, path_from_storage, resolve_directory
from stack_analyzer import (
    BC_CORR_NORM_TRC_COLUMN,
    MARKED_EVENTS_COLUMN,
    ROI_QUANT_PICKLE_NAME,
    SEGMENT_COLORS,
    format_quant_value_detail,
    load_quant_store,
    parse_freq_avr_field,
    parse_marked_events,
    parse_start_frames,
    summarize_quant_cell,
)

COLLECT_PICKLE_NAME = "pickle_stack_collect.pkl"
RESULTS_EXPORT_BASENAME = "results_figure.pdf"

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


def mean_sem(values) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(arr))
    if arr.size < 2:
        return mean, 0.0
    sem = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    return mean, sem


def iter_segment_peak_maxes(segments) -> list[float]:
    if not isinstance(segments, list):
        return []
    peaks: list[float] = []
    for segment in segments:
        if not isinstance(segment, np.ndarray) or segment.size == 0:
            continue
        valid = segment[np.isfinite(segment)]
        if valid.size:
            peaks.append(float(np.max(valid)))
    return peaks


def peaks_for_evoked_index(rows: list[dict], event_index: int) -> list[float]:
    peaks: list[float] = []
    for row in rows:
        evoked = row.get("Evoked Events")
        if not isinstance(evoked, list) or event_index >= len(evoked):
            continue
        segment = evoked[event_index]
        if not isinstance(segment, np.ndarray) or segment.size == 0:
            continue
        valid = segment[np.isfinite(segment)]
        if valid.size:
            peaks.append(float(np.max(valid)))
    return peaks


def pool_all_evoked_peak_maxes(rows: list[dict]) -> list[float]:
    peaks: list[float] = []
    for row in rows:
        peaks.extend(iter_segment_peak_maxes(row.get("Evoked Events")))
    return peaks


def pool_all_marked_peak_maxes(rows: list[dict]) -> list[float]:
    peaks: list[float] = []
    for row in rows:
        peaks.extend(iter_segment_peak_maxes(row.get("Marked Events")))
    return peaks


def pool_all_non_event_points(rows: list[dict]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for row in rows:
        non_events = row.get("Non-events")
        if not isinstance(non_events, list):
            continue
        for segment in non_events:
            if not isinstance(segment, np.ndarray) or segment.size == 0:
                continue
            valid = segment[np.isfinite(segment)]
            if valid.size:
                parts.append(valid.astype(np.float64, copy=False))
    if not parts:
        return np.array([], dtype=np.float64)
    return np.concatenate(parts)


def max_evoked_event_count(rows: list[dict]) -> int:
    max_count = 0
    for row in rows:
        evoked = row.get("Evoked Events")
        if isinstance(evoked, list):
            max_count = max(max_count, len(evoked))
    return max_count


def row_n_frames(row: dict) -> int:
    trace = row.get(BC_CORR_NORM_TRC_COLUMN)
    if trace is None:
        return 0
    return int(np.asarray(trace).size)


def row_event_mask(row: dict, n_frames: int) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    for start, end_exclusive in evoked_intervals_exclusive(row, n_frames):
        start = max(0, min(int(start), n_frames))
        end_exclusive = max(start, min(int(end_exclusive), n_frames))
        mask[start:end_exclusive] = True
    for start, end_inclusive in parse_marked_events(row.get(MARKED_EVENTS_COLUMN)):
        start = max(0, min(int(start), n_frames - 1))
        end_inclusive = max(start, min(int(end_inclusive), n_frames - 1))
        mask[start : end_inclusive + 1] = True
    return mask


def group_rows_by_experiment(
    rows: list[dict],
    collect_directory: str | Path,
    *,
    stored_base: str | Path | None = None,
) -> list[tuple[str, list[dict]]]:
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for row in rows:
        directory = row.get("directory")
        if directory:
            key = resolve_collect_directory_value(
                directory, collect_directory, stored_base=stored_base
            )
        else:
            key = "<unknown>"
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)
    return [(key, sorted(groups[key], key=lambda item: int(item.get("ROI #") or 0))) for key in order]


def experiment_display_names(keys: list[str]) -> dict[str, str]:
    """Use the first path segment where experiment directories differ."""
    if not keys:
        return {}
    if len(keys) == 1:
        key = keys[0]
        if not key or key == "<unknown>":
            return {key: key}
        parts = Path(key).parts
        return {key: parts[-1] if parts else key}

    parts_list = [Path(key).parts for key in keys]
    min_len = min(len(parts) for parts in parts_list)
    diff_index: int | None = None
    for index in range(min_len):
        if len({parts[index] for parts in parts_list}) > 1:
            diff_index = index
            break
    if diff_index is None and max(len(parts) for parts in parts_list) > min_len:
        diff_index = min_len

    labels: dict[str, str] = {}
    for key, parts in zip(keys, parts_list):
        if not key or key == "<unknown>":
            labels[key] = key
        elif diff_index is not None and diff_index < len(parts):
            labels[key] = parts[diff_index]
        elif parts:
            labels[key] = parts[-1]
        else:
            labels[key] = key
    return labels


def frame_axis_seconds(n_frames: int, freq_avr_text) -> np.ndarray:
    if n_frames <= 0:
        return np.array([], dtype=np.float64)
    try:
        fps, avr = parse_freq_avr_field(freq_avr_text)
    except (TypeError, ValueError):
        return np.arange(n_frames, dtype=np.float64)
    if fps <= 0 or avr <= 0:
        return np.arange(n_frames, dtype=np.float64)
    return np.arange(n_frames, dtype=np.float64) / (fps / avr)


RASTER_GREY = np.array([0.82, 0.82, 0.82])
RASTER_BLACK = np.array([0.08, 0.08, 0.08])
RASTER_RED = np.array([0.82, 0.18, 0.18])
COACTIVITY_OFF = np.array([0.93, 0.93, 0.93])
COACTIVITY_ON = np.array([0.12, 0.12, 0.12])


def build_experiment_raster(
    experiment_rows: list[dict], n_frames: int
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    n_rois = len(experiment_rows)
    img = np.tile(RASTER_GREY, (n_rois + 1, n_frames, 1))
    roi_masks: list[np.ndarray] = []

    for roi_idx, row in enumerate(experiment_rows):
        row_index = roi_idx + 1
        for start, end_exclusive in evoked_intervals_exclusive(row, n_frames):
            start = max(0, min(int(start), n_frames))
            end_exclusive = max(start, min(int(end_exclusive), n_frames))
            img[row_index, start:end_exclusive] = RASTER_BLACK
        for start, end_inclusive in parse_marked_events(row.get(MARKED_EVENTS_COLUMN)):
            start = max(0, min(int(start), n_frames - 1))
            end_inclusive = max(start, min(int(end_inclusive), n_frames - 1))
            img[row_index, start : end_inclusive + 1] = RASTER_RED
        roi_masks.append(row_event_mask(row, n_frames))

    if roi_masks:
        coactivity = np.any(np.stack(roi_masks, axis=0), axis=0)
    else:
        coactivity = np.zeros(n_frames, dtype=bool)
    img[0, coactivity] = COACTIVITY_ON
    img[0, ~coactivity] = COACTIVITY_OFF
    return img, experiment_rows, coactivity


def compute_timing_panel_data(
    rows: list[dict],
    collect_directory: str | Path,
    *,
    stored_base: str | Path | None = None,
) -> dict:
    grouped = group_rows_by_experiment(rows, collect_directory, stored_base=stored_base)
    labels = experiment_display_names([key for key, _rows in grouped])
    experiments: list[dict] = []
    for key, experiment_rows in grouped:
        n_frames = max((row_n_frames(row) for row in experiment_rows), default=0)
        if n_frames <= 0:
            continue
        _, _, coactivity = build_experiment_raster(experiment_rows, n_frames)
        experiments.append(
            {
                "key": key,
                "label": labels.get(key, key),
                "rows": experiment_rows,
                "n_rois": len(experiment_rows),
                "n_frames": n_frames,
                "coactivity_fraction": float(np.mean(coactivity)) if coactivity.size else 0.0,
            }
        )
    return {
        "experiments": experiments,
        "n_rows": len(rows),
        "n_experiments": len(experiments),
    }


MARKED_OVERLAY_COLOR = "#d81b60"


def pool_indexed_evoked_segments(
    rows: list[dict],
) -> list[tuple[np.ndarray, int, object]]:
    pooled: list[tuple[np.ndarray, int, object]] = []
    for row in rows:
        segments = row.get("Evoked Events")
        if not isinstance(segments, list):
            continue
        freq_avr = row.get("freq + avr")
        for event_index, segment in enumerate(segments):
            if isinstance(segment, np.ndarray) and segment.size:
                pooled.append((np.asarray(segment, dtype=np.float64), event_index, freq_avr))
    return pooled


def pool_marked_segments(rows: list[dict]) -> list[tuple[np.ndarray, object]]:
    pooled: list[tuple[np.ndarray, object]] = []
    for row in rows:
        segments = row.get("Marked Events")
        if not isinstance(segments, list):
            continue
        freq_avr = row.get("freq + avr")
        for segment in segments:
            if isinstance(segment, np.ndarray) and segment.size:
                pooled.append((np.asarray(segment, dtype=np.float64), freq_avr))
    return pooled


def segment_values_for_overlay(
    segment: np.ndarray,
    freq_avr_text,
    *,
    derivative: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    times = frame_axis_seconds(int(segment.size), freq_avr_text)
    if not derivative:
        return times, segment
    if times.size > 1:
        return times, np.gradient(segment, times)
    return times, np.zeros_like(segment)


def stack_segments_mean_seconds(
    segments: list[np.ndarray],
    freq_avr_texts: list,
) -> tuple[np.ndarray, np.ndarray]:
    if not segments:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    time_axes: list[np.ndarray] = []
    dts: list[float] = []
    for segment, freq_avr in zip(segments, freq_avr_texts):
        times = frame_axis_seconds(int(segment.size), freq_avr)
        time_axes.append(times)
        if times.size > 1:
            dts.append(float(times[1] - times[0]))
        elif times.size == 1:
            dts.append(1.0)

    dt = min(dts) if dts else 1.0
    t_max = max(float(times[-1]) for times in time_axes if times.size)
    grid = np.arange(0.0, t_max + dt / 2, dt, dtype=np.float64)

    stacked = np.full((len(segments), grid.size), np.nan, dtype=np.float64)
    for index, (segment, times) in enumerate(zip(segments, time_axes)):
        if times.size == 0:
            continue
        if times.size == 1:
            nearest = int(np.argmin(np.abs(grid - times[0])))
            stacked[index, nearest] = segment[0]
            continue
        valid = (grid >= times[0]) & (grid <= times[-1])
        if np.any(valid):
            stacked[index, valid] = np.interp(grid[valid], times, segment)
    return grid, np.nanmean(stacked, axis=0)


def compute_overlay_panel_data(rows: list[dict]) -> dict:
    evoked_segments = pool_indexed_evoked_segments(rows)
    marked_segments = pool_marked_segments(rows)
    return {
        "n_rows": len(rows),
        "n_evoked_segments": len(evoked_segments),
        "n_marked_segments": len(marked_segments),
    }


def draw_segment_overlay_ax(
    ax,
    segments: list[tuple[np.ndarray, object]],
    *,
    title: str,
    ylabel: str,
    color: str = MARKED_OVERLAY_COLOR,
    derivative: bool = False,
) -> None:
    ax.clear()
    if not segments:
        ax.text(
            0.5,
            0.5,
            "No segment data available.",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    plotted_segments: list[np.ndarray] = []
    freq_avr_texts: list[object] = []
    for segment, freq_avr in segments:
        times, values = segment_values_for_overlay(
            segment, freq_avr, derivative=derivative
        )
        plotted_segments.append(values)
        freq_avr_texts.append(freq_avr)
        ax.plot(
            times,
            values,
            color=color,
            linewidth=0.8,
            alpha=0.55,
            zorder=2,
        )

    x_mean, mean_values = stack_segments_mean_seconds(plotted_segments, freq_avr_texts)
    if x_mean.size:
        ax.plot(x_mean, mean_values, color="black", linewidth=2.4, zorder=5)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)


def draw_indexed_segment_overlay_ax(
    ax,
    indexed_segments: list[tuple[np.ndarray, int, object]],
    *,
    title: str,
    ylabel: str,
    derivative: bool = False,
) -> None:
    ax.clear()
    if not indexed_segments:
        ax.text(
            0.5,
            0.5,
            "No segment data available.",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    plotted_segments: list[np.ndarray] = []
    freq_avr_texts: list[object] = []
    for segment, event_index, freq_avr in indexed_segments:
        times, values = segment_values_for_overlay(
            segment, freq_avr, derivative=derivative
        )
        plotted_segments.append(values)
        freq_avr_texts.append(freq_avr)
        segment_color = SEGMENT_COLORS[event_index % len(SEGMENT_COLORS)]
        ax.plot(
            times,
            values,
            color=segment_color,
            linewidth=0.8,
            alpha=0.55,
            zorder=2,
        )

    x_mean, mean_values = stack_segments_mean_seconds(plotted_segments, freq_avr_texts)
    if x_mean.size:
        ax.plot(x_mean, mean_values, color="black", linewidth=2.4, zorder=5)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)


def segment_duration_seconds(row: dict, segment) -> float:
    """Convert a trace segment length to seconds using freq + avr."""
    if not isinstance(segment, np.ndarray) or segment.size == 0:
        return float("nan")
    text = row.get("freq + avr")
    if not text:
        return float("nan")
    try:
        fps, avr = parse_freq_avr_field(text)
    except (TypeError, ValueError):
        return float("nan")
    if fps <= 0 or avr <= 0:
        return float("nan")
    n_frames = int(segment.size)
    return n_frames / (fps / avr)


def iter_segment_durations(row: dict, segments) -> list[float]:
    if not isinstance(segments, list):
        return []
    durations: list[float] = []
    for segment in segments:
        duration = segment_duration_seconds(row, segment)
        if np.isfinite(duration):
            durations.append(duration)
    return durations


def durations_for_evoked_index(rows: list[dict], event_index: int) -> list[float]:
    durations: list[float] = []
    for row in rows:
        evoked = row.get("Evoked Events")
        if not isinstance(evoked, list) or event_index >= len(evoked):
            continue
        duration = segment_duration_seconds(row, evoked[event_index])
        if np.isfinite(duration):
            durations.append(duration)
    return durations


def pool_all_evoked_durations(rows: list[dict]) -> list[float]:
    durations: list[float] = []
    for row in rows:
        durations.extend(iter_segment_durations(row, row.get("Evoked Events")))
    return durations


def pool_all_marked_durations(rows: list[dict]) -> list[float]:
    durations: list[float] = []
    for row in rows:
        durations.extend(iter_segment_durations(row, row.get("Marked Events")))
    return durations


def compute_max_peak_panel_data(rows: list[dict]) -> dict:
    labels: list[str] = []
    means: list[float] = []
    sems: list[float] = []

    for event_index in range(max_evoked_event_count(rows)):
        peaks = peaks_for_evoked_index(rows, event_index)
        mean, sem = mean_sem(peaks)
        labels.append(f"Evoked {event_index + 1}")
        means.append(mean)
        sems.append(sem)

    all_evoked_peaks = pool_all_evoked_peak_maxes(rows)
    mean, sem = mean_sem(all_evoked_peaks)
    labels.append("All evoked")
    means.append(mean)
    sems.append(sem)

    marked_peaks = pool_all_marked_peak_maxes(rows)
    mean_marked, sem_marked = mean_sem(marked_peaks)
    labels.append("Spontaneous")
    means.append(mean_marked)
    sems.append(sem_marked)

    non_event_points = pool_all_non_event_points(rows)
    ref_mean, ref_sem = mean_sem(non_event_points)

    return {
        "labels": labels,
        "means": means,
        "sems": sems,
        "reference_mean": ref_mean,
        "reference_sem": ref_sem,
        "n_rows": len(rows),
        "n_evoked_peaks": len(all_evoked_peaks),
        "n_marked_peaks": len(marked_peaks),
        "n_non_event_points": int(non_event_points.size),
    }


def compute_duration_panel_data(rows: list[dict]) -> dict:
    labels: list[str] = []
    values: list[list[float]] = []
    means: list[float] = []
    sems: list[float] = []

    for event_index in range(max_evoked_event_count(rows)):
        durations = durations_for_evoked_index(rows, event_index)
        labels.append(f"Evoked {event_index + 1}")
        values.append(durations)
        mean, sem = mean_sem(durations)
        means.append(mean)
        sems.append(sem)

    all_evoked_durations = pool_all_evoked_durations(rows)
    labels.append("All evoked")
    values.append(all_evoked_durations)
    mean, sem = mean_sem(all_evoked_durations)
    means.append(mean)
    sems.append(sem)

    marked_durations = pool_all_marked_durations(rows)
    labels.append("Spontaneous")
    values.append(marked_durations)
    mean_marked, sem_marked = mean_sem(marked_durations)
    means.append(mean_marked)
    sems.append(sem_marked)

    return {
        "labels": labels,
        "values": values,
        "means": means,
        "sems": sems,
        "n_rows": len(rows),
        "n_evoked_durations": len(all_evoked_durations),
        "n_marked_durations": len(marked_durations),
    }


def max_peak_bar_colors(labels: list[str]) -> list[str]:
    """Match Mark Events segment colors for indexed evoked peaks."""
    colors: list[str] = []
    for label in labels:
        if label.startswith("Evoked ") and label != "All evoked":
            try:
                event_index = int(label.split()[-1]) - 1
            except ValueError:
                event_index = len(colors)
            colors.append(SEGMENT_COLORS[event_index % len(SEGMENT_COLORS)])
        else:
            colors.append("0.55")
    return colors


def focused_value_ylim(
    values: np.ndarray,
    errors: np.ndarray | None = None,
    *,
    reference: float | None = None,
    reference_error: float | None = None,
    padding_fraction: float = 0.12,
    min_padding: float = 0.01,
) -> tuple[float, float] | None:
    """Y-limits that zoom to data range (not forced to zero)."""
    lows: list[float] = []
    highs: list[float] = []

    values = np.asarray(values, dtype=np.float64)
    if errors is None:
        errors = np.zeros_like(values)
    else:
        errors = np.asarray(errors, dtype=np.float64)

    for value, err in zip(values, errors):
        if not np.isfinite(value):
            continue
        err = err if np.isfinite(err) else 0.0
        lows.append(float(value - err))
        highs.append(float(value + err))

    if reference is not None and np.isfinite(reference):
        ref_err = reference_error if reference_error is not None and np.isfinite(reference_error) else 0.0
        lows.append(float(reference - ref_err))
        highs.append(float(reference + ref_err))

    if not lows:
        return None

    y_min = min(lows)
    y_max = max(highs)
    span = y_max - y_min
    padding = max(span * padding_fraction, min_padding) if span > 0 else min_padding
    return y_min - padding, y_max + padding


def beeswarm_offsets(
    y_values: np.ndarray,
    *,
    x_width: float = 0.35,
    point_diameter: float = 0.07,
) -> np.ndarray:
    """Spread x offsets so overlapping points remain visible."""
    y = np.asarray(y_values, dtype=np.float64)
    n = y.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.zeros(1, dtype=np.float64)

    y_span = float(np.nanmax(y) - np.nanmin(y))
    y_tol = max(y_span * 0.015, 1e-6)
    order = np.argsort(y, kind="mergesort")
    offsets = np.zeros(n, dtype=np.float64)
    placed: list[tuple[float, float]] = []

    for idx in order:
        yi = float(y[idx])
        chosen = 0.0
        for step in range(64):
            candidates = (0.0,) if step == 0 else (step * point_diameter, -step * point_diameter)
            for cand in candidates:
                if abs(cand) > x_width:
                    break
                if all(
                    abs(cand - px) >= point_diameter or abs(yi - py) >= y_tol
                    for px, py in placed
                ):
                    chosen = cand
                    break
            else:
                continue
            break
        offsets[idx] = chosen
        placed.append((chosen, yi))
    return offsets


def build_overlay_panel_text(data: dict) -> str:
    return (
        f"(A) Evoked overlays and (B) their frame-wise derivatives show all "
        f"{data['n_evoked_segments']} pooled evoked segment trace(s) aligned to event onset, "
        "with thin lines colored by evoked position and a thick black mean trace. "
        f"(C) Spontaneous overlays and (D) their derivatives show all "
        f"{data['n_marked_segments']} pooled spontaneous segment(s) in magenta "
        "with the same mean overlay, from "
        f"{data['n_rows']} collected ROI row(s)."
    )


def build_max_peak_panel_text(data: dict) -> str:
    evoked_count = max(0, len(data["labels"]) - 2)
    evoked_clause = (
        f"Evoked 1–{evoked_count}"
        if evoked_count > 1
        else ("Evoked 1" if evoked_count == 1 else "no indexed evoked events")
    )
    return (
        "(E) Max Peak: mean maximum normalized BC-corrected trace value "
        f"from {data['n_rows']} collected ROI row(s). "
        f"Indexed evoked bars ({evoked_clause}) use one peak per ROI per event; "
        f"'All evoked' pools {data['n_evoked_peaks']} evoked segment peak(s); "
        f"'Spontaneous' pools {data['n_marked_peaks']} spontaneous segment peak(s). "
        "Error bars are SEM. The red dotted line and grey band show the global mean "
        f"± SEM across {data['n_non_event_points']} pooled non-event trace point(s)."
    )


def build_duration_panel_text(data: dict) -> str:
    evoked_count = max(0, len(data["labels"]) - 2)
    evoked_clause = (
        f"Evoked 1–{evoked_count}"
        if evoked_count > 1
        else ("Evoked 1" if evoked_count == 1 else "no indexed evoked events")
    )
    return (
        "(F) Duration: segment length converted to seconds as "
        "frames / (acquisition fps / averaging factor) using each row's "
        f"freq + avr field from {data['n_rows']} collected ROI row(s). "
        f"Indexed evoked groups ({evoked_clause}) use one duration per ROI per event; "
        f"'All evoked' pools {data['n_evoked_durations']} evoked segment(s); "
        f"'Spontaneous' pools {data['n_marked_durations']} spontaneous segment(s). "
        "Colored dots show individual segment durations; black horizontal lines with "
        "error bars indicate mean ± SEM within each category."
    )


def build_timing_panel_text(data: dict) -> str:
    if data["n_experiments"] == 0:
        return (
            "(G) Timing: no experiment groups with trace data were available for raster plots."
        )
    experiment_clause = (
        f"{data['n_experiments']} experiment/FOV group(s)"
        if data["n_experiments"] != 1
        else "one experiment/FOV group"
    )
    return (
        f"(G) Timing: {experiment_clause} from {data['n_rows']} collected ROI row(s), "
        "grouped by stack directory within the same field of view. "
        "Each raster row is one ROI on a grey background; black spans are evoked segments "
        "and red spans are spontaneous events. The top row shows co-activity "
        "(active when any ROI has an evoked or spontaneous event at that time), "
        "used to assess whether spontaneous events are local to one ROI or shared across ROIs."
    )


def build_results_figure_text(
    overlay_data: dict,
    peak_data: dict,
    duration_data: dict,
    timing_data: dict,
) -> str:
    return (
        f"Figure 1. {build_overlay_panel_text(overlay_data)} "
        f"{build_max_peak_panel_text(peak_data)} "
        f"{build_duration_panel_text(duration_data)} "
        f"{build_timing_panel_text(timing_data)}"
    )


def draw_category_mean_sem(
    ax,
    x_center: float,
    mean: float,
    sem: float,
    *,
    half_width: float = 0.18,
    zorder: int = 5,
) -> None:
    """Per-category mean indicator without connecting adjacent categories."""
    if not np.isfinite(mean):
        return
    ax.hlines(mean, x_center - half_width, x_center + half_width, colors="k", linewidth=1.8, zorder=zorder)
    if np.isfinite(sem) and sem > 0:
        ax.errorbar(
            [x_center],
            [mean],
            yerr=[sem],
            fmt="none",
            ecolor="k",
            elinewidth=1.2,
            capsize=5,
            capthick=1.5,
            zorder=zorder,
        )


def prompt_results_export_format(parent: tk.Misc) -> str | None:
    """Return export format key ('pdf') or None if cancelled."""
    choice: list[str | None] = [None]

    dialog = tk.Toplevel(parent)
    dialog.title("Export Results")
    dialog.transient(parent)
    dialog.grab_set()
    dialog.resizable(False, False)

    tk.Label(dialog, text="Format:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
    format_var = tk.StringVar(value="PDF (.pdf)")
    format_combo = ttk.Combobox(
        dialog,
        textvariable=format_var,
        values=["PDF (.pdf)"],
        state="readonly",
        width=20,
    )
    format_combo.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

    button_frame = tk.Frame(dialog)
    button_frame.grid(row=1, column=0, columnspan=2, pady=(0, 10))

    def accept() -> None:
        selected = format_var.get()
        if selected.startswith("PDF"):
            choice[0] = "pdf"
        dialog.destroy()

    def cancel() -> None:
        dialog.destroy()

    tk.Button(button_frame, text="Export", width=10, command=accept).pack(side="left", padx=4)
    tk.Button(button_frame, text="Cancel", width=10, command=cancel).pack(side="left", padx=4)
    dialog.bind("<Escape>", lambda _event: cancel())
    dialog.protocol("WM_DELETE_WINDOW", cancel)
    dialog.wait_window()
    return choice[0]


def resolve_results_export_path(parent: tk.Misc, default_path: Path) -> Path | None:
    """Resolve export path, prompting to overwrite or rename if needed."""
    if not default_path.exists():
        return default_path

    response = messagebox.askyesnocancel(
        "File exists",
        f"{default_path.name} already exists in:\n{default_path.parent}\n\n"
        "Choose Yes to overwrite, No to rename, or Cancel.",
        parent=parent,
    )
    if response is None:
        return None
    if response:
        return default_path

    renamed = filedialog.asksaveasfilename(
        parent=parent,
        title="Export Results As",
        initialdir=str(default_path.parent),
        initialfile=f"{default_path.stem}_copy{default_path.suffix}",
        defaultextension=default_path.suffix,
        filetypes=[("PDF files", "*.pdf")],
    )
    if not renamed:
        return None
    return Path(renamed)


def estimate_caption_height_inches(caption: str, *, figure_width: float) -> float:
    chars_per_line = max(70, int(figure_width * 10.5))
    line_count = 0
    for paragraph in caption.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            line_count += 1
            continue
        line_count += len(textwrap.wrap(paragraph, width=chars_per_line)) or 1
    return min(4.5, max(1.1, 0.2 * line_count + 0.45))


def attach_figure_caption(fig: Figure, caption: str) -> dict:
    """Reserve space below plots for caption text during export."""
    axes = list(fig.axes)
    width, height = fig.get_size_inches()
    caption_height = estimate_caption_height_inches(caption, figure_width=width)
    gap_height = 0.25
    new_height = height + caption_height + gap_height

    state = {
        "size": (float(width), float(height)),
        "axes": axes,
        "positions": [ax.get_position().bounds for ax in axes],
    }

    fig.set_size_inches(width, new_height, forward=True)
    caption_fraction = caption_height / new_height
    gap_fraction = gap_height / new_height
    plot_scale = height / new_height

    for ax in axes:
        pos = ax.get_position()
        ax.set_position(
            [
                pos.x0,
                (pos.y0 * plot_scale) + caption_fraction + gap_fraction,
                pos.width,
                pos.height * plot_scale,
            ]
        )

    caption_ax = fig.add_axes([0.05, 0.01, 0.9, max(0.08, caption_fraction - 0.01)])
    caption_ax.set_axis_off()
    caption_ax.text(
        0.0,
        1.0,
        caption,
        transform=caption_ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        wrap=True,
    )
    state["caption_ax"] = caption_ax
    return state


def detach_figure_caption(fig: Figure, state: dict) -> None:
    state["caption_ax"].remove()
    fig.set_size_inches(state["size"][0], state["size"][1], forward=True)
    for ax, bounds in zip(state["axes"], state["positions"]):
        ax.set_position(bounds)


class ResultsWindow:
    """Results figure window for the collected dataset."""

    def __init__(self, app: "StackAnalyzerTotalApp") -> None:
        rows = list(app.store.get("rows", []))
        if not rows:
            messagebox.showinfo(
                "Results",
                "No collected rows available. Add experiments and rebuild the collection first.",
            )
            return

        self.app = app
        app._results_window = self
        self.rows = rows

        self.root = tk.Toplevel(app.root)
        self.root.title("Results")
        self.root.geometry("1560x1180")
        self.root.minsize(1180, 860)

        plot_frame = tk.Frame(self.root)
        plot_frame.pack(fill="both", expand=True, padx=8, pady=8)

        self.fig = Figure(figsize=(15.5, 11.0), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        caption_frame = tk.LabelFrame(self.root, text="Figure Text", padx=8, pady=8)
        caption_frame.pack(fill="x", padx=8, pady=(0, 8))
        self.caption_text = tk.Text(caption_frame, height=11, wrap="word")
        self.caption_text.pack(fill="x")
        self.caption_text.config(state="disabled")

        button_frame = tk.Frame(self.root)
        button_frame.pack(fill="x", padx=8, pady=(0, 8))
        tk.Button(button_frame, text="Export…", command=self._export_figure).pack(side="right")

        self._draw_figure(self.rows)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _export_figure(self) -> None:
        export_format = prompt_results_export_format(self.root)
        if export_format is None:
            return
        if export_format != "pdf":
            messagebox.showerror("Export", f"Unsupported export format: {export_format}", parent=self.root)
            return

        default_path = self.app.collect_path.parent / RESULTS_EXPORT_BASENAME
        export_path = resolve_results_export_path(self.root, default_path)
        if export_path is None:
            return

        caption = self.caption_text.get("1.0", "end-1c").strip()
        caption_state = None
        try:
            if caption:
                caption_state = attach_figure_caption(self.fig, caption)
            self.fig.savefig(export_path, format="pdf", pad_inches=0.2)
        except OSError as exc:
            messagebox.showerror("Export failed", str(exc), parent=self.root)
            return
        finally:
            if caption_state is not None:
                detach_figure_caption(self.fig, caption_state)
                self.fig.tight_layout()
                self.canvas.draw_idle()

        messagebox.showinfo("Export", f"Saved results figure to:\n{export_path}", parent=self.root)

    def _on_close(self) -> None:
        self.app._results_window = None
        self.root.destroy()

    def _set_caption(self, caption: str) -> None:
        self.caption_text.config(state="normal")
        self.caption_text.delete("1.0", "end")
        self.caption_text.insert("1.0", caption)
        self.caption_text.config(state="disabled")

    def _draw_figure(self, rows: list[dict]) -> None:
        timing_data = compute_timing_panel_data(
            rows,
            self.app.collect_directory,
            stored_base=self.app.stored_collect_directory,
        )
        timing_height = max(1.0, timing_data["n_experiments"] * 0.9)

        self.fig.clear()
        gs = self.fig.add_gridspec(
            3,
            2,
            width_ratios=[1.15, 1.0],
            height_ratios=[1.0, 1.0, timing_height],
            wspace=0.34,
            hspace=0.42,
        )
        gs_left = gs[:, 0].subgridspec(2, 2, hspace=0.35, wspace=0.28)
        ax_evoked = self.fig.add_subplot(gs_left[0, 0])
        ax_evoked_deriv = self.fig.add_subplot(gs_left[0, 1])
        ax_marked = self.fig.add_subplot(gs_left[1, 0])
        ax_marked_deriv = self.fig.add_subplot(gs_left[1, 1])

        ax_peak = self.fig.add_subplot(gs[0, 1])
        ax_duration = self.fig.add_subplot(gs[1, 1])
        if timing_data["n_experiments"] > 0:
            gs_timing = gs[2, 1].subgridspec(timing_data["n_experiments"], 1, hspace=0.55)
            timing_axes = [
                self.fig.add_subplot(gs_timing[i]) for i in range(timing_data["n_experiments"])
            ]
        else:
            timing_axes = [self.fig.add_subplot(gs[2, 1])]

        overlay_data = self._draw_overlay_on_axes(
            ax_evoked,
            ax_evoked_deriv,
            ax_marked,
            ax_marked_deriv,
            rows,
        )
        peak_data = self._draw_max_peak_on_ax(ax_peak, rows)
        duration_data = self._draw_duration_on_ax(ax_duration, rows)
        timing_data = self._draw_timing_on_axes(timing_axes, timing_data)
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self._set_caption(
            build_results_figure_text(overlay_data, peak_data, duration_data, timing_data)
        )

    def _draw_overlay_on_axes(
        self,
        ax_evoked,
        ax_evoked_deriv,
        ax_marked,
        ax_marked_deriv,
        rows: list[dict],
    ) -> dict:
        data = compute_overlay_panel_data(rows)
        evoked_segments = pool_indexed_evoked_segments(rows)
        marked_segments = pool_marked_segments(rows)

        draw_indexed_segment_overlay_ax(
            ax_evoked,
            evoked_segments,
            title="(A) Evoked",
            ylabel="Normalized trace",
        )
        draw_indexed_segment_overlay_ax(
            ax_evoked_deriv,
            evoked_segments,
            title="(B) Evoked derivative",
            ylabel="d(trace)/d(time)",
            derivative=True,
        )
        draw_segment_overlay_ax(
            ax_marked,
            marked_segments,
            title="(C) Spontaneous",
            ylabel="Normalized trace",
        )
        draw_segment_overlay_ax(
            ax_marked_deriv,
            marked_segments,
            title="(D) Spontaneous derivative",
            ylabel="d(trace)/d(time)",
            derivative=True,
        )
        return data

    def _draw_max_peak_on_ax(self, ax, rows: list[dict]) -> dict:
        data = compute_max_peak_panel_data(rows)
        ax.clear()

        labels = data["labels"]
        means = np.asarray(data["means"], dtype=np.float64)
        sems = np.asarray(data["sems"], dtype=np.float64)

        ref_mean = data["reference_mean"]
        ref_sem = data["reference_sem"]

        if labels and np.any(np.isfinite(means)):
            x = np.arange(len(labels))
            valid_sem = np.where(np.isfinite(sems), sems, 0.0)
            y_limits = focused_value_ylim(
                means,
                valid_sem,
                reference=ref_mean,
                reference_error=ref_sem,
            )
            bar_bottom = y_limits[0] if y_limits is not None else 0.0
            bar_colors = max_peak_bar_colors(labels)
            ax.bar(
                x,
                means - bar_bottom,
                bottom=bar_bottom,
                yerr=valid_sem,
                capsize=4,
                color=bar_colors,
                edgecolor=bar_colors,
                linewidth=0.8,
                zorder=3,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30 if len(labels) > 4 else 0, ha="right")
            ax.set_ylabel("Max peak (normalized)")
            if y_limits is not None:
                ax.set_ylim(y_limits)
        else:
            ax.text(
                0.5,
                0.5,
                "No evoked or spontaneous segment data available.",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

        if np.isfinite(ref_mean):
            ax.axhline(ref_mean, color="red", linestyle=":", linewidth=1.5, zorder=4)
            if np.isfinite(ref_sem) and ref_sem > 0:
                ax.axhspan(
                    ref_mean - ref_sem,
                    ref_mean + ref_sem,
                    color="0.75",
                    alpha=0.35,
                    zorder=1,
                )

        ax.set_title("(E) Max Peak")
        return data

    def _draw_duration_on_ax(self, ax, rows: list[dict]) -> dict:
        data = compute_duration_panel_data(rows)
        ax.clear()

        labels = data["labels"]
        groups = data["values"]
        means = np.asarray(data["means"], dtype=np.float64)
        sems = np.asarray(data["sems"], dtype=np.float64)

        has_values = any(len(group) > 0 for group in groups)
        if labels and has_values:
            x = np.arange(len(labels))
            colors = max_peak_bar_colors(labels)
            all_values: list[float] = []

            for xi, (group, color) in enumerate(zip(groups, colors)):
                arr = np.asarray(group, dtype=np.float64)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    continue
                all_values.extend(arr.tolist())
                offsets = beeswarm_offsets(arr)
                ax.scatter(
                    xi + offsets,
                    arr,
                    s=30,
                    color=color,
                    edgecolor="0.25",
                    linewidth=0.5,
                    alpha=0.9,
                    zorder=3,
                )

            valid_sem = np.where(np.isfinite(sems), sems, 0.0)
            finite_means = np.where(np.isfinite(means), means, np.nan)
            for xi, mean, sem in zip(x, finite_means, valid_sem):
                draw_category_mean_sem(ax, float(xi), float(mean), float(sem))

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30 if len(labels) > 4 else 0, ha="right")
            ax.set_ylabel("Duration (s)")

            if all_values:
                y_limits = focused_value_ylim(np.asarray(all_values, dtype=np.float64))
                if y_limits is not None:
                    ax.set_ylim(y_limits)
        else:
            ax.text(
                0.5,
                0.5,
                "No evoked or spontaneous segment data available.",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

        ax.set_title("(F) Duration")
        return data

    def _draw_timing_on_axes(self, axes, timing_data: dict) -> dict:
        experiments = timing_data["experiments"]
        if not experiments:
            ax = axes[0]
            ax.clear()
            ax.text(
                0.5,
                0.5,
                "No experiment groups with trace data available.",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_title("(G) Timing")
            ax.set_xticks([])
            ax.set_yticks([])
            return timing_data

        for ax, experiment in zip(axes, experiments):
            ax.clear()
            experiment_rows = experiment["rows"]
            n_frames = int(experiment["n_frames"])
            img, sorted_rows, _coactivity = build_experiment_raster(experiment_rows, n_frames)
            times = frame_axis_seconds(n_frames, experiment_rows[0].get("freq + avr"))
            if times.size:
                if times.size > 1:
                    dt = float(times[1] - times[0])
                else:
                    dt = 1.0
                extent = [float(times[0] - dt / 2), float(times[-1] + dt / 2), len(sorted_rows) + 1, 0]
            else:
                extent = [0, 1, len(sorted_rows) + 1, 0]

            ax.imshow(
                img,
                aspect="auto",
                interpolation="nearest",
                origin="upper",
                extent=extent,
            )
            ax.axhline(1.0, color="0.35", linewidth=0.8, zorder=2)

            y_labels = ["Co-activity"] + [
                f"ROI {row.get('ROI #', idx + 1)}" for idx, row in enumerate(sorted_rows)
            ]
            y_ticks = np.arange(len(y_labels)) + 0.5
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels)
            ax.set_title(experiment["label"], fontsize=9)
            if ax is axes[-1]:
                ax.set_xlabel("Time (s)")
            else:
                ax.set_xticklabels([])

        axes[0].text(
            0.0,
            1.12,
            "(G) Timing",
            transform=axes[0].transAxes,
            fontsize=11,
            fontweight="bold",
            va="bottom",
        )
        axes[0].plot([], [], color="0.08", linewidth=6, label="Evoked")
        axes[0].plot([], [], color=(0.82, 0.18, 0.18), linewidth=6, label="Spontaneous")
        axes[0].legend(
            loc="upper right",
            bbox_to_anchor=(1.0, 1.12),
            fontsize=8,
            frameon=False,
            handlelength=1.2,
        )
        return timing_data


class StackAnalyzerTotalApp:
    def __init__(self, initial_dir: str | None = None) -> None:
        self.collect_directory = (
            str(Path(initial_dir).resolve()) if initial_dir else str(Path.cwd().resolve())
        )
        self.stored_collect_directory = self.collect_directory
        self.store = empty_collect_store(self.collect_directory)
        self.collect_path = collect_pickle_path_for_directory(self.collect_directory)
        self._results_window: ResultsWindow | None = None

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
        tk.Button(button_col, text="Results", width=16, command=self._on_results).pack()

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

    def _on_results(self) -> None:
        if self._results_window is not None:
            try:
                if self._results_window.root.winfo_exists():
                    self._results_window.root.lift()
                    self._results_window.root.focus_force()
                    return
            except (tk.TclError, AttributeError):
                pass
            self._results_window = None

        if not self.collect_path.exists():
            messagebox.showinfo(
                "Results",
                f"No collection file found at:\n{self.collect_path}\n\nRebuild the collection first.",
            )
            return
        if not self.store.get("rows"):
            messagebox.showinfo(
                "Results",
                "No collected rows available. Add experiments and rebuild the collection.",
            )
            return
        ResultsWindow(self)

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
