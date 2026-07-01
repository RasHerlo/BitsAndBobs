#!/usr/bin/env python3
"""Interactive GUI for TIFF stack ROI fluorescence trace analysis."""

from __future__ import annotations

import argparse
import os
import pickle
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from matplotlib import colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import tifffile
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon, Rectangle
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

from portable_paths import directory_matches, resolve_directory

SEGMENT_COLORS = plt.cm.tab10.colors
HANDLE_RADIUS = 14
MIN_FREEHAND_POINTS = 4
MIN_POINT_SPACING = 1.5
MAX_EDIT_VERTICES = 64
DEFAULT_STARTS = "896, 1050, 1205, 1359, 1513"
DEFAULT_ACQ_FPS = 20.548
DEFAULT_AVR = 4
ROI_QUANT_PICKLE_NAME = "ROI_quant pickle.pkl"

QUANT_COLUMNS = [
    "directory",
    "size",
    "starts",
    "freq + avr",
    "SG window and order",
    "extension",
    "Area L+R",
    "BG pixels",
    "BG trc",
    "ROI pixels",
    "ROI trc",
    "Area",
    "max vals",
    "bleach correct",
    "BC baseline",
    "fit params",
    "BC auto shift",
    "Man. Adj.",
    "Marked events",
    "BC-corr. Norm-Trc",
]

MARKED_EVENTS_COLUMN = "Marked events"
BC_CORR_NORM_TRC_COLUMN = "BC-corr. Norm-Trc"
FIT_PARAMS_COLUMN = "fit params"
BC_AUTO_SHIFT_COLUMN = "BC auto shift"


def quant_pickle_path_for_stack(stack_path: str) -> Path:
    return Path(stack_path).resolve().parent / ROI_QUANT_PICKLE_NAME


def empty_quant_store() -> dict:
    return {"version": 1, "columns": QUANT_COLUMNS, "rows": []}


def load_quant_store(path: Path) -> dict:
    if not path.exists():
        return empty_quant_store()
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict) or "rows" not in data:
        return empty_quant_store()
    data["columns"] = list(QUANT_COLUMNS)
    for row in data.get("rows", []):
        for column in QUANT_COLUMNS:
            if column not in row:
                row[column] = None
    return data


def save_quant_store(path: Path, store: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(store, handle, protocol=pickle.HIGHEST_PROTOCOL)


def ensure_quant_pickle(stack_path: str) -> Path:
    path = quant_pickle_path_for_stack(stack_path)
    if not path.exists():
        save_quant_store(path, empty_quant_store())
    return path


def compute_max_vals_per_segment(
    normalized_segments: list[np.ndarray],
    rel_x: np.ndarray,
    f_left: int,
    f_right: int,
) -> list[float]:
    if f_right < f_left:
        f_left, f_right = f_right, f_left
    mask = (rel_x >= f_left) & (rel_x <= f_right)
    if not np.any(mask):
        return [float("nan")] * len(normalized_segments)

    max_vals: list[float] = []
    for segment in normalized_segments:
        region = segment[mask]
        valid = region[~np.isnan(region)]
        max_vals.append(float(np.max(valid)) if valid.size else float("nan"))
    return max_vals


def format_stack_size(width: int, height: int, n_frames: int) -> str:
    return f"{width}x{height}x{n_frames}"


def parse_freq_avr_field(text: str) -> tuple[float, float]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) < 2:
        raise ValueError(f"Invalid freq + avr field: {text!r}")
    fps = float(parts[0].lower().replace("fps", "").strip())
    avr = float(parts[1].lower().replace("x", "").strip())
    return fps, avr


def format_freq_avr_field(acq_fps: float, avr_factor: float) -> str:
    avr_display = (
        f"{int(avr_factor)}x" if float(avr_factor).is_integer() else f"{avr_factor}x"
    )
    return f"{acq_fps} fps, {avr_display}"


def parse_sg_field(text: str) -> tuple[int, int]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) < 2:
        raise ValueError(f"Invalid SG window and order field: {text!r}")
    return int(float(parts[0])), int(float(parts[1]))


def format_sg_field(window: int, poly: int) -> str:
    return f"{int(window)}, {int(poly)}"


def parse_area_lr_field(text: str) -> tuple[int, int]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) < 2:
        raise ValueError(f"Invalid Area L+R field: {text!r}")
    left, right = int(float(parts[0])), int(float(parts[1]))
    if right < left:
        left, right = right, left
    return left, right


def format_area_lr_field(left: int, right: int) -> str:
    if right < left:
        left, right = right, left
    return f"{left}, {right}"


def quant_settings_from_row(row: dict) -> dict:
    fps, avr = parse_freq_avr_field(row["freq + avr"])
    sg_window, sg_poly = parse_sg_field(row["SG window and order"])
    area_left, area_right = parse_area_lr_field(row["Area L+R"])
    extension_raw = row.get("extension", "50")
    return {
        "starts": str(row["starts"]),
        "acq_fps": fps,
        "avr_factor": avr,
        "sg_window": sg_window,
        "sg_poly": sg_poly,
        "extension": int(float(extension_raw)),
        "area_left": area_left,
        "area_right": area_right,
    }


def quant_settings_to_row_fields(settings: dict) -> dict:
    return {
        "starts": settings["starts"],
        "freq + avr": format_freq_avr_field(settings["acq_fps"], settings["avr_factor"]),
        "SG window and order": format_sg_field(settings["sg_window"], settings["sg_poly"]),
        "extension": str(int(settings["extension"])),
        "Area L+R": format_area_lr_field(settings["area_left"], settings["area_right"]),
    }


def quant_settings_equal(left: dict, right: dict) -> bool:
    if left["starts"].replace(" ", "") != right["starts"].replace(" ", ""):
        return False
    if abs(left["acq_fps"] - right["acq_fps"]) > 1e-6:
        return False
    if abs(left["avr_factor"] - right["avr_factor"]) > 1e-6:
        return False
    if left["sg_window"] != right["sg_window"] or left["sg_poly"] != right["sg_poly"]:
        return False
    if left["extension"] != right["extension"]:
        return False
    return left["area_left"] == right["area_left"] and left["area_right"] == right["area_right"]


TRACE_SUMMARY_COLUMNS = frozenset(
    {"BG trc", "ROI trc", "bleach correct", "BC baseline", BC_CORR_NORM_TRC_COLUMN}
)


def biexponential_decay(t: np.ndarray, a1: float, tau1: float, a2: float, tau2: float, c: float) -> np.ndarray:
    return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + c


def format_fit_params(a1: float, tau1: float, a2: float, tau2: float, c: float) -> str:
    return ", ".join(f"{float(value):.6g}" for value in (a1, tau1, a2, tau2, c))


def parse_fit_params(value) -> tuple[float, float, float, float, float] | None:
    if value is None:
        return None
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if len(parts) < 5:
        return None
    try:
        return tuple(float(parts[index]) for index in range(5))  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def parse_bc_auto_shift(value, *, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def format_bc_auto_shift(enabled: bool) -> int:
    return 1 if enabled else 0


def bleach_correct_from_fit_params(n_frames: int, params: tuple[float, float, float, float, float]) -> np.ndarray:
    t = np.arange(int(n_frames), dtype=np.float64)
    fitted = biexponential_decay(t, *params)
    return np.asarray(fitted, dtype=np.float64)


def fit_biexponential_params(signal: np.ndarray) -> tuple[float, float, float, float, float] | None:
    """Fit f(t) = A1*exp(-t/tau1) + A2*exp(-t/tau2) + C to the full trace."""
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or values.size < 6:
        return None
    if not np.all(np.isfinite(values)):
        return None

    t = np.arange(values.size, dtype=np.float64)
    tail = float(values[-1])
    head = float(values[0])
    amplitude = head - tail
    if abs(amplitude) < 1e-12:
        amplitude = float(np.ptp(values)) or 1.0
    span = max(float(values.size), 1.0)
    initial = [
        amplitude * 0.6,
        max(span / 10.0, 1.0),
        amplitude * 0.4,
        max(span / 2.0, 1.0),
        tail,
    ]

    try:
        params, _ = curve_fit(
            biexponential_decay,
            t,
            values,
            p0=initial,
            maxfev=50_000,
        )
    except (RuntimeError, ValueError, TypeError):
        return None

    fitted = biexponential_decay(t, *params)
    if not np.all(np.isfinite(fitted)):
        return None
    return tuple(float(value) for value in params)


def fit_biexponential_bleach_correct(signal: np.ndarray) -> np.ndarray | None:
    params = fit_biexponential_params(signal)
    if params is None:
        return None
    return bleach_correct_from_fit_params(len(signal), params)


def compute_bc_baseline_for_smooth(
    smooth: np.ndarray,
    row: dict | None = None,
) -> np.ndarray | None:
    """Build BC baseline from stored fit params when available, else auto-fit."""
    smooth = np.asarray(smooth, dtype=np.float64)
    if smooth.size == 0:
        return None

    params = parse_fit_params(row.get(FIT_PARAMS_COLUMN)) if row is not None else None
    if params is not None:
        bleach_correct = bleach_correct_from_fit_params(smooth.size, params)
    else:
        bleach_correct = fit_biexponential_bleach_correct(smooth)
    if bleach_correct is None:
        return None

    auto_shift = parse_bc_auto_shift(
        row.get(BC_AUTO_SHIFT_COLUMN) if row is not None else None,
        default=True,
    )
    if auto_shift:
        return compute_bc_baseline_trace(bleach_correct, smooth)
    return bleach_correct.astype(np.float64, copy=True)


def compute_bg_corrected_smooth_trace(
    roi_trace: np.ndarray,
    bg_trace: np.ndarray,
    sg_window: int,
    sg_poly: int,
) -> np.ndarray:
    roi = np.asarray(roi_trace, dtype=np.float64)
    bg = np.asarray(bg_trace, dtype=np.float64)
    length = min(len(roi), len(bg))
    roi = roi[:length]
    bg = bg[:length]
    smooth_roi = apply_savgol(roi, sg_window, sg_poly)
    smooth_bg = apply_savgol(bg, sg_window, sg_poly)
    return smooth_roi - smooth_bg


def compute_bc_baseline_trace(
    bleach_correct: np.ndarray,
    bg_corrected_smooth: np.ndarray,
    below_fraction: float = 0.10,
) -> np.ndarray:
    """Parallel-shift bleach-correct down so `below_fraction` of smooth values lie below it."""
    bleach = np.asarray(bleach_correct, dtype=np.float64)
    smooth = np.asarray(bg_corrected_smooth, dtype=np.float64)
    length = min(len(bleach), len(smooth))
    bleach = bleach[:length]
    smooth = smooth[:length]
    gap = bleach - smooth
    percentile = min(max((1.0 - below_fraction) * 100.0, 0.0), 100.0)
    shift = float(np.percentile(gap, percentile))
    return (bleach - shift).astype(np.float64)


def row_needs_bleach_update(row: dict) -> bool:
    if row.get("BG trc") is None:
        return False
    for column in ("bleach correct", "BC baseline", FIT_PARAMS_COLUMN):
        value = row.get(column)
        if value is None:
            return True
        if isinstance(value, np.ndarray) and value.size == 0:
            return True
    return False


def update_row_bleach_correction(row: dict, *, force_auto_fit: bool = False) -> bool:
    """Compute bleach correct and BC baseline for one pickle row. Returns True if filled."""
    bg_trace = row.get("BG trc")
    roi_trace = row.get("ROI trc")
    if bg_trace is None or roi_trace is None:
        row["bleach correct"] = None
        row["BC baseline"] = None
        row[FIT_PARAMS_COLUMN] = None
        row[BC_CORR_NORM_TRC_COLUMN] = None
        return False

    try:
        sg_window, sg_poly = parse_sg_field(row["SG window and order"])
    except (ValueError, KeyError, TypeError):
        row["bleach correct"] = None
        row["BC baseline"] = None
        row[FIT_PARAMS_COLUMN] = None
        row[BC_CORR_NORM_TRC_COLUMN] = None
        return False

    if row.get(BC_AUTO_SHIFT_COLUMN) is None:
        row[BC_AUTO_SHIFT_COLUMN] = format_bc_auto_shift(True)

    smooth = compute_bg_corrected_smooth_trace(roi_trace, bg_trace, sg_window, sg_poly)
    params = None if force_auto_fit else parse_fit_params(row.get(FIT_PARAMS_COLUMN))
    if params is None:
        params = fit_biexponential_params(smooth)
    if params is None:
        row["bleach correct"] = None
        row["BC baseline"] = None
        row[FIT_PARAMS_COLUMN] = None
        row[BC_CORR_NORM_TRC_COLUMN] = None
        return False

    bleach_correct = bleach_correct_from_fit_params(smooth.size, params)
    row[FIT_PARAMS_COLUMN] = format_fit_params(*params)
    auto_shift = parse_bc_auto_shift(row.get(BC_AUTO_SHIFT_COLUMN))
    if auto_shift:
        bc_baseline = compute_bc_baseline_trace(bleach_correct, smooth)
    else:
        bc_baseline = bleach_correct.astype(np.float64, copy=True)

    row["bleach correct"] = bleach_correct
    row["BC baseline"] = bc_baseline
    update_row_bc_corr_norm_trc(row)
    return True


def update_row_fit_taus(row: dict, tau1: float, tau2: float) -> bool:
    params = parse_fit_params(row.get(FIT_PARAMS_COLUMN))
    if params is None:
        if not update_row_bleach_correction(row, force_auto_fit=True):
            return False
        params = parse_fit_params(row.get(FIT_PARAMS_COLUMN))
        if params is None:
            return False
    updated = (float(params[0]), float(tau1), float(params[2]), float(tau2), float(params[4]))
    row[FIT_PARAMS_COLUMN] = format_fit_params(*updated)
    return update_row_bleach_correction(row, force_auto_fit=False)


def update_row_bc_auto_shift(row: dict, enabled: bool) -> bool:
    row[BC_AUTO_SHIFT_COLUMN] = format_bc_auto_shift(enabled)
    if parse_fit_params(row.get(FIT_PARAMS_COLUMN)) is None:
        return update_row_bleach_correction(row, force_auto_fit=True)
    return update_row_bleach_correction(row, force_auto_fit=False)


def update_row_bc_corr_norm_trc(row: dict, man_adj: float | None = None) -> bool:
    """Compute and store the Mark Events normalized trace for one pickle row."""
    if man_adj is None:
        man_adj = parse_man_adj(row.get("Man. Adj."))
    _smooth, _baseline, normalized = compute_row_mark_event_traces(row, man_adj)
    if normalized is None:
        row[BC_CORR_NORM_TRC_COLUMN] = None
        return False
    row[BC_CORR_NORM_TRC_COLUMN] = np.asarray(normalized, dtype=np.float64)
    return True


def row_needs_norm_trc_update(row: dict) -> bool:
    if row.get("BG trc") is None:
        return False
    value = row.get(BC_CORR_NORM_TRC_COLUMN)
    if value is None:
        return True
    if isinstance(value, np.ndarray) and value.size == 0:
        return True
    return False


def backfill_bc_corr_norm_trc(store: dict) -> bool:
    changed = False
    for row in store.get("rows", []):
        if not row_needs_norm_trc_update(row):
            continue
        if update_row_bc_corr_norm_trc(row):
            changed = True
    return changed


def backfill_bleach_correction(store: dict) -> bool:
    """Fill missing bleach fields for rows that have a BG trace. Returns True if any row changed."""
    changed = False
    for row in store.get("rows", []):
        if not row_needs_bleach_update(row):
            continue
        if update_row_bleach_correction(row):
            changed = True
    return changed


def parse_man_adj(value) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def ensure_man_adj_defaults(store: dict) -> bool:
    changed = False
    for row in store.get("rows", []):
        if row.get("Man. Adj.") is None:
            row["Man. Adj."] = 0.0
            changed = True
        if row.get(BC_AUTO_SHIFT_COLUMN) is None:
            row[BC_AUTO_SHIFT_COLUMN] = format_bc_auto_shift(True)
            changed = True
    return changed


def compute_row_mark_event_traces(
    row: dict,
    man_adj: float = 0.0,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Return smoothed BG-corrected trace, shifted BC baseline, and normalized trace."""
    bg_trace = row.get("BG trc")
    roi_trace = row.get("ROI trc")
    if bg_trace is None or roi_trace is None:
        return None, None, None

    if row_needs_bleach_update(row):
        update_row_bleach_correction(row)

    bc_baseline = row.get("BC baseline")
    if bc_baseline is None:
        return None, None, None

    try:
        sg_window, sg_poly = parse_sg_field(row["SG window and order"])
    except (ValueError, KeyError, TypeError):
        return None, None, None

    smooth = compute_bg_corrected_smooth_trace(roi_trace, bg_trace, sg_window, sg_poly)
    length = min(len(smooth), len(bc_baseline))
    if length == 0:
        return None, None, None

    smooth = smooth[:length]
    adjusted_baseline = np.asarray(bc_baseline[:length], dtype=np.float64) + man_adj

    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = smooth / adjusted_baseline
    normalized = np.where(np.isfinite(normalized), normalized, np.nan)

    return smooth, adjusted_baseline, normalized


def draw_row_event_spans(
    ax,
    row: dict,
    n_frames: int,
    frame_to_axis: Callable[[float], float],
) -> None:
    """Draw colored event brackets from a pickle row's starts and extension."""
    try:
        extension = int(float(row.get("extension", "50")))
    except (TypeError, ValueError):
        extension = 50
    starts = parse_start_frames(str(row.get("starts", "")), n_frames)
    for idx, start in enumerate(starts):
        if start + extension > n_frames:
            continue
        color = SEGMENT_COLORS[idx % len(SEGMENT_COLORS)]
        span_left = frame_to_axis(float(start))
        span_right = frame_to_axis(float(start + extension))
        ax.axvspan(span_left, span_right, color=color, alpha=0.18, lw=0)


def parse_marked_events(value) -> list[tuple[int, int]]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        if arr.size == 0:
            return []
        if arr.ndim == 1 and arr.size >= 2:
            arr = arr.reshape(1, -1)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return _normalize_event_pairs([(int(r[0]), int(r[1])) for r in arr])
        return []
    if isinstance(value, list):
        pairs: list[tuple[int, int]] = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                pairs.append((int(item[0]), int(item[1])))
        return _normalize_event_pairs(pairs)
    return []


def _normalize_event_pairs(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    normalized: list[tuple[int, int]] = []
    for start, finish in pairs:
        if finish < start:
            start, finish = finish, start
        normalized.append((start, finish))
    return normalized


def format_marked_events_for_storage(pairs: list[tuple[int, int]]) -> list[list[int]]:
    return [[int(start), int(finish)] for start, finish in _normalize_event_pairs(pairs)]


def bg_pixels_equal(left, right) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    if left_arr.shape != right_arr.shape:
        return False
    return bool(np.allclose(left_arr, right_arr, atol=1e-6, rtol=0.0))


def draw_locked_marked_events(
    ax,
    row: dict,
    n_frames: int,
    frame_to_axis: Callable[[float], float],
) -> None:
    for start, finish in parse_marked_events(row.get(MARKED_EVENTS_COLUMN)):
        if start >= n_frames:
            continue
        finish = min(finish, n_frames - 1)
        span_left = frame_to_axis(float(start))
        span_right = frame_to_axis(float(finish))
        ax.axvspan(span_left, span_right, color="0.55", alpha=0.22, lw=0, zorder=2)


def summarize_quant_cell(
    column: str,
    value,
    *,
    current_directory: str | Path | None = None,
) -> str:
    """Short one-line value for the pickle inspector table."""
    if value is None:
        return ""
    if column == "directory" and current_directory is not None:
        value = resolve_directory(str(value), current_directory)
    if column == MARKED_EVENTS_COLUMN:
        pairs = parse_marked_events(value)
        if not pairs:
            return ""
        if len(pairs) <= 3:
            return "; ".join(f"{start}-{finish}" for start, finish in pairs)
        return f"{len(pairs)} events"
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return "[]"
        if value.ndim == 2 and value.shape[1] == 2:
            return f"{value.shape[0]} vertices"
        if value.ndim == 1:
            if column.endswith("trc") or column in TRACE_SUMMARY_COLUMNS:
                vmin = float(np.nanmin(value))
                vmax = float(np.nanmax(value))
                return f"{value.size} pts ({vmin:.3g}…{vmax:.3g})"
            if column == "max vals":
                if value.size <= 6:
                    return np.array2string(value, precision=3, separator=", ")
                return f"{value.size} values"
            return f"array[{value.size}]"
        return f"array{value.shape}"
    text = str(value)
    if column == "directory" and len(text) > 48:
        return text[:45] + "…"
    return text


def format_quant_value_detail(
    column: str,
    value,
    *,
    current_directory: str | Path | None = None,
) -> str:
    """Multi-line detail for one pickle row field."""
    if value is None:
        return "None"
    if column == "directory" and current_directory is not None:
        value = resolve_directory(str(value), current_directory)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return "[]"
        if value.ndim == 2 and value.shape[1] == 2:
            return np.array2string(value, precision=2, separator=", ", max_line_width=100)
        if value.ndim == 1 and value.size > 24 and (
            column.endswith("trc") or column in TRACE_SUMMARY_COLUMNS
        ):
            stats = (
                f"length={value.size}, "
                f"min={float(np.nanmin(value)):.6g}, "
                f"max={float(np.nanmax(value)):.6g}, "
                f"mean={float(np.nanmean(value)):.6g}"
            )
            head = np.array2string(value[:12], precision=4, separator=", ")
            tail = np.array2string(value[-6:], precision=4, separator=", ")
            return f"{stats}\n{head} … ({value.size} total) … {tail}"
        return np.array2string(value, precision=4, separator=", ", max_line_width=100)
    return str(value)


def format_quant_row_detail(
    row: dict,
    row_index: int,
    columns: list[str],
    *,
    current_directory: str | Path | None = None,
) -> str:
    lines = [f"Row {row_index}"]
    for column in columns:
        if column not in row:
            continue
        lines.append(f"\n{column}:")
        lines.append(
            format_quant_value_detail(
                column,
                row[column],
                current_directory=current_directory,
            )
        )
    return "\n".join(lines)


def open_quant_pickle_inspector(
    path: Path,
    *,
    current_directory: str | Path | None = None,
) -> None:
    """Open a scrollable overview of the ROI quantification pickle file."""
    store = load_quant_store(path)
    columns = list(store.get("columns", QUANT_COLUMNS))
    rows = store.get("rows", [])
    resolved_directory = (
        Path(current_directory).resolve()
        if current_directory is not None
        else path.parent.resolve()
    )
    tree_columns = ["#"] + columns

    root = tk.Tk()
    root.withdraw()

    window = tk.Toplevel(root)
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
        text=f"version={store.get('version', '?')}  |  {len(rows)} row(s)",
        anchor="w",
    ).pack(fill="x")

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
        "size": 110,
        "starts": 150,
        "freq + avr": 110,
        "SG window and order": 120,
        "extension": 70,
        "Area L+R": 80,
        "BG pixels": 90,
        "BG trc": 130,
        "ROI pixels": 90,
        "ROI trc": 130,
        "Area": 90,
        "max vals": 120,
        "bleach correct": 130,
        "BC baseline": 130,
        "fit params": 150,
        "BC auto shift": 90,
        "Man. Adj.": 70,
        "Marked events": 120,
        "BC-corr. Norm-Trc": 130,
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
                summarize_quant_cell(
                    column,
                    row.get(column),
                    current_directory=resolved_directory,
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
                format_quant_row_detail(
                    rows[row_index],
                    row_index,
                    columns,
                    current_directory=resolved_directory,
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
        root.destroy()

    window.protocol("WM_DELETE_WINDOW", on_close)


def load_tif_stack(path: str) -> np.ndarray:
    """Load a TIFF stack as (frames, height, width)."""
    with tifffile.TiffFile(path) as tif:
        stack = tif.asarray()

    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    elif stack.ndim != 3:
        raise ValueError(f"Expected 2D or 3D stack, got shape {stack.shape}")

    return stack


def compute_z_average(stack: np.ndarray) -> np.ndarray:
    return stack.mean(axis=0)


def compute_raw_trace(stack: np.ndarray, mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None or not np.any(mask):
        return None

    return stack[:, mask].mean(axis=1)


def apply_savgol(trace: np.ndarray, window: int, polyorder: int, axis: int = 0) -> np.ndarray:
    window = int(window)
    if window % 2 == 0:
        window += 1
    length = trace.shape[axis]
    max_window = length - 1 if length % 2 == 0 else length
    window = max(3, min(window, max_window))
    polyorder = min(int(polyorder), window - 1)
    if window < 3 or polyorder < 1:
        return trace.copy()
    return savgol_filter(trace, window_length=window, polyorder=polyorder, axis=axis)


def compute_area_from_mean_trace(
    rel_x: np.ndarray,
    mean_values: np.ndarray,
    f_left: int,
    f_right: int,
    baseline_level: float = 1.0,
) -> np.ndarray | float:
    """Integrate (mean - baseline) over relative frames. Supports 1D or 2D mean_values."""
    if f_right < f_left:
        f_left, f_right = f_right, f_left

    squeeze = mean_values.ndim == 1
    if squeeze:
        mean_values = mean_values[:, np.newaxis]

    frames = rel_x.astype(float)
    values = mean_values - baseline_level
    overlap = (frames >= f_left) & (frames <= f_right)
    if not np.any(overlap):
        result = np.zeros(mean_values.shape[1], dtype=np.float64)
        return float(result[0]) if squeeze else result

    x = frames[overlap]
    y = values[overlap, :]

    if f_left < x[0]:
        y0 = np.array([np.interp(f_left, frames, values[:, col]) for col in range(values.shape[1])])
        x = np.concatenate([[f_left], x])
        y = np.concatenate([y0[np.newaxis, :], y], axis=0)
    if f_right > x[-1]:
        y1 = np.array([np.interp(f_right, frames, values[:, col]) for col in range(values.shape[1])])
        x = np.concatenate([x, [f_right]])
        y = np.concatenate([y, y1[np.newaxis, :]], axis=0)

    areas = trapezoid(y, x, axis=0)
    return float(areas[0]) if squeeze else areas


def compute_all_pixel_mean_traces(
    stack: np.ndarray,
    starts: list[int],
    extension: int,
    window: int,
    polyorder: int,
    baseline_fraction: float = 0.2,
    progress: Callable[[str, float], None] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return relative x axis and mean normalized traces with shape (total_len, H, W)."""
    n_frames, height, width = stack.shape
    n_pixels = height * width
    pixels = stack.reshape(n_frames, -1).astype(np.float64)

    def report(stage: str, fraction: float) -> None:
        if progress is not None:
            progress(stage, fraction)

    report("Preparing stack", 0.0)
    chunk_size = max(4096, n_pixels // 40)
    smooth = np.empty_like(pixels)

    for col_start in range(0, n_pixels, chunk_size):
        col_end = min(col_start + chunk_size, n_pixels)
        smooth[:, col_start:col_end] = apply_savgol(
            pixels[:, col_start:col_end], window, polyorder, axis=0
        )
        report("Smoothing pixels", 0.05 + 0.50 * (col_end / n_pixels))

    baseline_len, total_len, rel_x = segment_geometry(extension, baseline_fraction)
    valid_starts = [start for start in starts if start + extension <= n_frames]
    if not valid_starts:
        return None

    aligned_segments: list[np.ndarray] = []
    n_starts = len(valid_starts)
    for idx, start in enumerate(valid_starts):
        seg_start = max(0, start - baseline_len)
        raw = smooth[seg_start : start + extension, :]
        available_baseline = start - seg_start
        baseline_mean = raw[:available_baseline, :].mean(axis=0)
        baseline_mean = np.where(baseline_mean == 0, 1.0, baseline_mean)

        aligned = np.full((total_len, raw.shape[1]), np.nan, dtype=np.float64)
        offset = baseline_len - available_baseline
        aligned[offset : offset + raw.shape[0], :] = raw / baseline_mean[np.newaxis, :]
        aligned_segments.append(aligned)
        report("Building segments", 0.55 + 0.35 * ((idx + 1) / n_starts))

    report("Averaging segments", 0.92)
    mean_trace = np.nanmean(np.stack(aligned_segments, axis=0), axis=0)
    mean_trace = mean_trace.reshape(total_len, height, width)
    report("Pixel traces ready", 1.0)
    return rel_x, mean_trace


def compute_pixel_area_map(
    stack: np.ndarray,
    starts: list[int],
    extension: int,
    window: int,
    polyorder: int,
    f_left: int,
    f_right: int,
    baseline_fraction: float = 0.2,
) -> np.ndarray | None:
    """Per-pixel area values using the same pipeline as the ROI mean trace."""
    result = compute_all_pixel_mean_traces(
        stack, starts, extension, window, polyorder, baseline_fraction
    )
    if result is None:
        return None

    rel_x, mean_trace = result
    flat_areas = compute_area_from_mean_trace(
        rel_x, mean_trace.reshape(mean_trace.shape[0], -1), f_left, f_right
    )
    return np.asarray(flat_areas, dtype=np.float64).reshape(mean_trace.shape[1], mean_trace.shape[2])


def parse_start_frames(text: str, n_frames: int) -> list[int]:
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    if not parts:
        return [0]

    values = [max(0, min(int(float(p)), n_frames - 1)) for p in parts]
    return values


def parse_start_seconds(text: str, n_frames: int, fps: float) -> list[int]:
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    if not parts:
        return [0]
    if fps <= 0:
        return parse_start_frames(text, n_frames)

    values = [max(0, min(int(round(float(p) * fps)), n_frames - 1)) for p in parts]
    return values


def format_start_frames(frames: list[int]) -> str:
    return ", ".join(str(int(f)) for f in frames)


def format_start_seconds(frames: list[int], fps: float) -> str:
    if fps <= 0:
        return format_start_frames(frames)
    seconds = [int(round(f / fps)) for f in frames]
    return ", ".join(str(s) for s in seconds)


def segment_geometry(extension: int, baseline_fraction: float = 0.2) -> tuple[int, int, np.ndarray]:
    """Return baseline length, total relative length, and 1-based relative frame axis."""
    baseline_len = max(1, int(round(baseline_fraction * extension)))
    total_len = baseline_len + extension
    rel_x = np.arange(1, total_len + 1)
    return baseline_len, total_len, rel_x


def build_normalized_segments(
    smooth: np.ndarray,
    starts: list[int],
    extension: int,
    baseline_fraction: float = 0.2,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, list[int], int] | None:
    """Align, normalize (baseline -> 1), and overlay segments on a 1-based relative axis."""
    baseline_len, total_len, rel_x = segment_geometry(extension, baseline_fraction)
    normalized_segments: list[np.ndarray] = []
    valid_starts: list[int] = []

    for start in starts:
        if start + extension > len(smooth):
            continue

        seg_start = max(0, start - baseline_len)
        raw = smooth[seg_start : start + extension]
        available_baseline = start - seg_start
        baseline_mean = float(raw[:available_baseline].mean())
        if baseline_mean == 0:
            baseline_mean = 1.0

        aligned = np.full(total_len, np.nan)
        offset = baseline_len - available_baseline
        aligned[offset : offset + len(raw)] = raw / baseline_mean
        normalized_segments.append(aligned)
        valid_starts.append(start)

    if not normalized_segments:
        return None

    seg_array = np.array(normalized_segments)
    mean_normalized = np.nanmean(seg_array, axis=0)
    return rel_x, normalized_segments, mean_normalized, valid_starts, baseline_len


def polygon_mask(vertices: np.ndarray, height: int, width: int) -> np.ndarray:
    """Build a boolean image mask from polygon vertices in (x=col, y=row) coordinates."""
    yy, xx = np.mgrid[0:height, 0:width]
    points = np.column_stack([xx.ravel(), yy.ravel()])
    return MplPath(vertices).contains_points(points).reshape(height, width)


def clip_vertices(vertices: np.ndarray, width: int, height: int) -> np.ndarray:
    clipped = vertices.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
    return clipped


class MarkEventsWindow:
    """Secondary window for inspecting saved ROIs and adjusting BC baseline shift."""

    def __init__(self, app: "StackAnalyzerApp") -> None:
        if app.stack is None or app.z_average is None or app.quant_pickle_path is None:
            messagebox.showinfo("Mark Events", "Load a TIFF stack first.")
            return

        self.app = app
        self.store = load_quant_store(app.quant_pickle_path)
        store_changed = ensure_man_adj_defaults(self.store)
        if backfill_bleach_correction(self.store):
            store_changed = True
        if backfill_bc_corr_norm_trc(self.store):
            store_changed = True
        if store_changed:
            save_quant_store(app.quant_pickle_path, self.store)

        current_dir = os.path.dirname(os.path.abspath(app.file_path))
        current_size = format_stack_size(
            app.stack.shape[2], app.stack.shape[1], app.stack.shape[0]
        )
        self.entries: list[tuple[int, dict]] = []
        for row_index, row in enumerate(self.store.get("rows", [])):
            if not directory_matches(row.get("directory"), current_dir):
                continue
            if row.get("size") != current_size:
                continue
            self.entries.append((row_index, row))

        if not self.entries:
            messagebox.showinfo(
                "Mark Events",
                "No saved ROI rows for this stack in the pickle file.",
            )
            return

        app._mark_events_window = self
        self.entry_pos = 0
        if app._active_saved_roi_row_index is not None:
            for pos, (row_index, _row) in enumerate(self.entries):
                if row_index == app._active_saved_roi_row_index:
                    self.entry_pos = pos
                    break

        self._man_adj_step = 1.0
        self._updating_man_adj_field = False
        self._tau1_step = 1.0
        self._tau2_step = 1.0
        self._updating_tau_fields = False
        self._add_event_active = False
        self._remove_event_active = False
        self._draft_start: int | None = None
        self._draft_finish: int | None = None
        self._draft_phase = "idle"
        self._dragging_boundary: str | None = None
        self._canvas_cids: list[int] = []

        self.root = tk.Toplevel()
        self.root.title("Mark Events")
        self.root.geometry("1520x760")
        self.root.minsize(1100, 580)

        top = tk.Frame(self.root, padx=8, pady=6)
        top.pack(fill="x")

        tk.Button(top, text="◀", width=3, command=self._prev_row).pack(side="left")
        self.row_label = tk.Label(top, text="", font=("", 11, "bold"))
        self.row_label.pack(side="left", padx=10)
        tk.Button(top, text="▶", width=3, command=self._next_row).pack(side="left")

        top_controls = tk.Frame(top, padx=12)
        top_controls.pack(side="right")

        fit_frame = tk.Frame(top_controls)
        fit_frame.pack(side="left", padx=(0, 10))

        self._build_tau_controls(fit_frame, "τ₁", "tau1")
        self._build_tau_controls(fit_frame, "τ₂", "tau2", pad_left=10)

        self.bc_auto_shift_var = tk.BooleanVar(value=True)
        self._updating_bc_auto_shift = False
        self.bc_auto_shift_check = tk.Checkbutton(
            fit_frame,
            text="BC auto shift",
            variable=self.bc_auto_shift_var,
            command=self._on_bc_auto_shift_changed,
        )
        self.bc_auto_shift_check.pack(side="left", padx=(10, 6))

        tk.Button(fit_frame, text="Reset fit", command=self._reset_fit).pack(side="left", padx=(4, 0))

        adj_frame = tk.Frame(top_controls)
        adj_frame.pack(side="left")
        tk.Label(adj_frame, text="Man. Adj.").pack(side="left", padx=(0, 6))
        tk.Button(adj_frame, text="▼", width=2, command=lambda: self._bump_man_adj(-1)).pack(
            side="left"
        )
        self.man_adj_var = tk.DoubleVar(value=0.0)
        self.man_adj_entry = tk.Entry(
            adj_frame, textvariable=self.man_adj_var, width=10, justify="center"
        )
        self.man_adj_entry.pack(side="left", padx=4)
        self.man_adj_entry.bind("<Return>", self._on_man_adj_commit)
        self.man_adj_entry.bind("<FocusOut>", self._on_man_adj_commit)
        self.man_adj_var.trace_add("write", self._on_man_adj_live)
        tk.Button(adj_frame, text="▲", width=2, command=lambda: self._bump_man_adj(1)).pack(
            side="left"
        )

        content = tk.Frame(self.root)
        content.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        plot_frame = tk.Frame(content)
        plot_frame.pack(side="left", fill="both", expand=True)

        side_panel = tk.Frame(content, padx=8, pady=8)
        side_panel.pack(side="right", fill="y")
        self.btn_add_event = tk.Button(
            side_panel, text="Add Event", width=12, command=self._on_add_event
        )
        self.btn_add_event.pack(pady=(0, 6))
        self.btn_lock_event = tk.Button(
            side_panel, text="Lock Event", width=12, command=self._on_lock_event
        )
        self.btn_lock_event.pack(pady=(0, 6))
        self.btn_remove_event = tk.Button(
            side_panel, text="Remove Event", width=12, command=self._on_remove_event
        )
        self.btn_remove_event.pack()
        self.event_status_label = tk.Label(
            side_panel, text="", wraplength=120, justify="left", anchor="w"
        )
        self.event_status_label.pack(pady=(12, 0), fill="x")

        self.fig = Figure(figsize=(12.0, 7.0), dpi=100)
        grid = self.fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.28)
        self.ax_image = self.fig.add_subplot(grid[0, 0])
        right = grid[0, 1].subgridspec(2, 1, hspace=0.34)
        self.ax_upper = self.fig.add_subplot(right[0, 0])
        self.ax_lower = self.fig.add_subplot(right[1, 0], sharex=self.ax_upper)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._connect_canvas_events()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._refresh_all()

    def _on_close(self) -> None:
        for cid in self._canvas_cids:
            try:
                self.canvas.mpl_disconnect(cid)
            except (ValueError, AttributeError):
                pass
        self._canvas_cids = []
        self.app._mark_events_window = None
        self.root.destroy()

    def _connect_canvas_events(self) -> None:
        self._canvas_cids = [
            self.canvas.mpl_connect("button_press_event", self._on_canvas_press),
            self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion),
            self.canvas.mpl_connect("button_release_event", self._on_canvas_release),
        ]

    def _reset_event_modes(self) -> None:
        self._add_event_active = False
        self._remove_event_active = False
        self._draft_start = None
        self._draft_finish = None
        self._draft_phase = "idle"
        self._dragging_boundary = None
        self._sync_add_event_button()
        self._sync_remove_event_button()
        self._set_event_status("")

    def _set_event_status(self, message: str) -> None:
        self.event_status_label.config(text=message)

    def _sync_add_event_button(self) -> None:
        label = "Add Event [ON]" if self._add_event_active else "Add Event"
        self.btn_add_event.config(text=label)

    def _sync_remove_event_button(self) -> None:
        label = "Remove Event [ON]" if self._remove_event_active else "Remove Event"
        self.btn_remove_event.config(text=label)

    def _format_event_range(self, start: int, finish: int) -> str:
        app = self.app
        if app.convert_time_axis:
            fps = app._effective_fps()
            if fps > 0:
                return f"{start / fps:.3f}–{finish / fps:.3f} s"
        return f"{start}–{finish}"

    def _pick_locked_event_index(self, xdata: float) -> int | None:
        frame = self._clamp_frame(self._axis_to_frame(xdata))
        pairs = parse_marked_events(self._current_row().get(MARKED_EVENTS_COLUMN))
        for idx in range(len(pairs) - 1, -1, -1):
            start, finish = pairs[idx]
            if start <= frame <= finish:
                return idx
        return None

    def _current_n_frames(self) -> int:
        row = self._current_row()
        roi_trace = row.get("ROI trc")
        if roi_trace is not None:
            return len(np.asarray(roi_trace))
        return self.app.n_frames

    def _clamp_frame(self, frame: int) -> int:
        n_frames = self._current_n_frames()
        if n_frames <= 0:
            return 0
        return int(np.clip(frame, 0, n_frames - 1))

    def _axis_to_frame(self, axis_value: float) -> int:
        app = self.app
        if not app.convert_time_axis:
            return int(round(axis_value))
        fps = app._effective_fps()
        if fps <= 0:
            return int(round(axis_value))
        return int(round(axis_value * fps))

    def _axis_pick_tolerance(self, ax) -> float:
        try:
            bbox = ax.get_window_extent(self.canvas.get_renderer())
        except (AttributeError, RuntimeError):
            bbox = None
        if bbox is None or bbox.width <= 0:
            return 0.5
        xlim = ax.get_xlim()
        return abs(xlim[1] - xlim[0]) * (8.0 / bbox.width)

    def _pick_boundary(self, xdata: float) -> str | None:
        tol = self._axis_pick_tolerance(self.ax_lower)
        app = self.app
        if self._draft_start is not None:
            x_start = app._frame_to_axis(float(self._draft_start), one_based=False)
            if abs(xdata - x_start) <= tol:
                return "start"
        if self._draft_finish is not None:
            x_finish = app._frame_to_axis(float(self._draft_finish), one_based=False)
            if abs(xdata - x_finish) <= tol:
                return "finish"
        return None

    def _on_add_event(self) -> None:
        self._remove_event_active = False
        self._sync_remove_event_button()
        self._add_event_active = True
        self._draft_start = None
        self._draft_finish = None
        self._draft_phase = "need_start"
        self._dragging_boundary = None
        self._sync_add_event_button()
        self._set_event_status("Click the lower trace to set event start.")
        self._draw_traces()

    def _on_lock_event(self) -> None:
        if self._draft_start is None or self._draft_finish is None:
            messagebox.showinfo(
                "Lock Event",
                "Add an event first: set start and finish on the lower trace.",
            )
            return

        start = self._clamp_frame(self._draft_start)
        finish = self._clamp_frame(self._draft_finish)
        if finish < start:
            start, finish = finish, start

        row_index = self._current_row_index()
        pairs = parse_marked_events(self.store["rows"][row_index].get(MARKED_EVENTS_COLUMN))
        pairs.append((start, finish))
        self.store["rows"][row_index][MARKED_EVENTS_COLUMN] = format_marked_events_for_storage(
            pairs
        )
        save_quant_store(self.app.quant_pickle_path, self.store)
        self._reset_event_modes()
        self._draw_traces()
        self._set_event_status(f"Locked event {start}–{finish}.")

    def _on_remove_event(self) -> None:
        pairs = parse_marked_events(self._current_row().get(MARKED_EVENTS_COLUMN))
        if not pairs:
            messagebox.showinfo("Remove Event", "No locked events to remove for this row.")
            return

        self._reset_event_modes()
        self._remove_event_active = True
        self._sync_remove_event_button()
        self._set_event_status("Click a grey event region on the lower trace to remove it.")
        self._draw_traces()

    def _remove_locked_event_at_index(self, event_index: int) -> None:
        row_index = self._current_row_index()
        pairs = parse_marked_events(self.store["rows"][row_index].get(MARKED_EVENTS_COLUMN))
        if event_index < 0 or event_index >= len(pairs):
            return

        start, finish = pairs[event_index]
        label = self._format_event_range(start, finish)
        if not messagebox.askyesno("Remove Event", f"Delete locked event {label}?"):
            return

        pairs.pop(event_index)
        self.store["rows"][row_index][MARKED_EVENTS_COLUMN] = format_marked_events_for_storage(
            pairs
        )
        save_quant_store(self.app.quant_pickle_path, self.store)
        self._draw_traces()
        if pairs:
            self._set_event_status(f"Removed event {label}. Click another grey region to remove.")
        else:
            self._remove_event_active = False
            self._sync_remove_event_button()
            self._set_event_status(f"Removed event {label}. No locked events remain.")

    def _on_canvas_press(self, event) -> None:
        if event.inaxes is not self.ax_lower or event.xdata is None:
            return

        xdata = float(event.xdata)

        if self._remove_event_active:
            event_index = self._pick_locked_event_index(xdata)
            if event_index is not None:
                self._remove_locked_event_at_index(event_index)
            return

        if self._draft_start is not None and self._draft_finish is not None:
            boundary = self._pick_boundary(xdata)
            if boundary is not None:
                self._dragging_boundary = boundary
                return

        if not self._add_event_active:
            return

        frame = self._clamp_frame(self._axis_to_frame(xdata))
        if self._draft_phase == "need_start":
            self._draft_start = frame
            self._draft_phase = "need_finish"
            self._set_event_status("Click the lower trace to set event finish.")
            self._draw_traces()
        elif self._draft_phase == "need_finish":
            self._draft_finish = frame
            if self._draft_start is not None and self._draft_finish < self._draft_start:
                self._draft_start, self._draft_finish = self._draft_finish, self._draft_start
            self._draft_phase = "ready"
            self._add_event_active = False
            self._sync_add_event_button()
            self._set_event_status("Adjust boundaries, then click Lock Event.")
            self._draw_traces()

    def _on_canvas_motion(self, event) -> None:
        if self._dragging_boundary is None:
            return
        if event.inaxes is not self.ax_lower or event.xdata is None:
            return

        frame = self._clamp_frame(self._axis_to_frame(float(event.xdata)))
        if self._dragging_boundary == "start":
            if self._draft_finish is not None:
                frame = min(frame, self._draft_finish)
            self._draft_start = frame
        elif self._dragging_boundary == "finish":
            if self._draft_start is not None:
                frame = max(frame, self._draft_start)
            self._draft_finish = frame
        self._draw_traces()

    def _on_canvas_release(self, _event) -> None:
        self._dragging_boundary = None

    def _draw_draft_boundaries(self, n_frames: int, frame_to_axis: Callable[[float], float]) -> None:
        if self._draft_start is not None:
            self.ax_lower.axvline(
                frame_to_axis(float(self._draft_start)),
                color="black",
                linestyle=":",
                linewidth=1.5,
                zorder=6,
            )
        if self._draft_finish is not None:
            self.ax_lower.axvline(
                frame_to_axis(float(self._draft_finish)),
                color="black",
                linestyle=":",
                linewidth=1.5,
                zorder=6,
            )

    def _current_row_index(self) -> int:
        return self.entries[self.entry_pos][0]

    def _current_row(self) -> dict:
        return self.entries[self.entry_pos][1]

    def _build_tau_controls(self, parent: tk.Frame, label: str, attr: str, *, pad_left: int = 0) -> None:
        frame = tk.Frame(parent)
        frame.pack(side="left", padx=(pad_left, 0))
        tk.Label(frame, text=label).pack(side="left", padx=(0, 4))
        tk.Button(
            frame,
            text="▼",
            width=2,
            command=lambda: self._bump_tau(attr, -1),
        ).pack(side="left")
        var = tk.DoubleVar(value=1.0)
        entry = tk.Entry(frame, textvariable=var, width=8, justify="center")
        entry.pack(side="left", padx=3)
        entry.bind("<Return>", self._on_tau_commit)
        entry.bind("<FocusOut>", self._on_tau_commit)
        var.trace_add("write", self._on_tau_live)
        tk.Button(
            frame,
            text="▲",
            width=2,
            command=lambda: self._bump_tau(attr, 1),
        ).pack(side="left")
        setattr(self, f"{attr}_var", var)
        setattr(self, f"{attr}_entry", entry)

    def _tau_attr_index(self, attr: str) -> int:
        return 1 if attr == "tau1" else 3

    def _current_tau_value(self, attr: str) -> float:
        var = getattr(self, f"{attr}_var")
        try:
            return float(var.get())
        except (tk.TclError, ValueError):
            params = parse_fit_params(self._current_row().get(FIT_PARAMS_COLUMN))
            if params is None:
                return 1.0
            return float(params[self._tau_attr_index(attr)])

    def _tau_step_for_attr(self, attr: str) -> float:
        return self._tau1_step if attr == "tau1" else self._tau2_step

    def _bump_tau(self, attr: str, direction: int) -> None:
        current = self._current_tau_value(attr)
        self._set_tau_var(attr, current + direction * self._tau_step_for_attr(attr))
        self._save_fit_params_to_pickle()

    def _on_tau_live(self, *_args) -> None:
        if self._updating_tau_fields:
            return
        try:
            float(self.tau1_var.get())
            float(self.tau2_var.get())
        except (tk.TclError, ValueError):
            return
        self._draw_traces()
        self.canvas.draw_idle()

    def _on_tau_commit(self, _event=None) -> None:
        self._save_fit_params_to_pickle()

    def _set_tau_var(self, attr: str, value: float) -> None:
        self._updating_tau_fields = True
        getattr(self, f"{attr}_var").set(value)
        self._updating_tau_fields = False

    def _sync_fit_controls_from_row(self, row: dict) -> None:
        params = parse_fit_params(row.get(FIT_PARAMS_COLUMN))
        self._updating_tau_fields = True
        if params is None:
            self.tau1_var.set(1.0)
            self.tau2_var.set(1.0)
        else:
            self.tau1_var.set(float(params[1]))
            self.tau2_var.set(float(params[3]))
        self._updating_tau_fields = False

        self._updating_bc_auto_shift = True
        self.bc_auto_shift_var.set(parse_bc_auto_shift(row.get(BC_AUTO_SHIFT_COLUMN)))
        self._updating_bc_auto_shift = False

        if params is not None:
            self._tau1_step = max(1e-6, abs(float(params[1])) * 0.02)
            self._tau2_step = max(1e-6, abs(float(params[3])) * 0.02)
        else:
            self._tau1_step = 1.0
            self._tau2_step = 1.0

    def _save_fit_params_to_pickle(self) -> None:
        if self._updating_tau_fields:
            return
        try:
            tau1 = float(self.tau1_var.get())
            tau2 = float(self.tau2_var.get())
        except (tk.TclError, ValueError):
            row = self._current_row()
            self._sync_fit_controls_from_row(row)
            return

        row_index = self._current_row_index()
        row = self.store["rows"][row_index]
        if not update_row_fit_taus(row, tau1, tau2):
            return
        save_quant_store(self.app.quant_pickle_path, self.store)
        self._draw_traces()
        self.canvas.draw_idle()

    def _on_bc_auto_shift_changed(self) -> None:
        if self._updating_bc_auto_shift:
            return
        row_index = self._current_row_index()
        row = self.store["rows"][row_index]
        update_row_bc_auto_shift(row, bool(self.bc_auto_shift_var.get()))
        save_quant_store(self.app.quant_pickle_path, self.store)
        self._draw_traces()
        self.canvas.draw_idle()

    def _reset_fit(self) -> None:
        row_index = self._current_row_index()
        row = self.store["rows"][row_index]
        if not update_row_bleach_correction(row, force_auto_fit=True):
            messagebox.showinfo(
                "Reset fit",
                "Could not re-fit bleach correction for this row.",
                parent=self.root,
            )
            return
        save_quant_store(self.app.quant_pickle_path, self.store)
        self._sync_fit_controls_from_row(row)
        self._draw_traces()
        self.canvas.draw_idle()

    def _prev_row(self) -> None:
        if self.entry_pos > 0:
            self.entry_pos -= 1
            self._refresh_all()

    def _next_row(self) -> None:
        if self.entry_pos < len(self.entries) - 1:
            self.entry_pos += 1
            self._refresh_all()

    def _bump_man_adj(self, direction: int) -> None:
        current = self._current_man_adj_value()
        self.man_adj_var.set(current + direction * self._man_adj_step)
        self._save_man_adj_to_pickle()

    def _on_man_adj_live(self, *_args) -> None:
        if self._updating_man_adj_field:
            return
        try:
            float(self.man_adj_var.get())
        except (tk.TclError, ValueError):
            return
        self._draw_traces()
        self.canvas.draw_idle()

    def _on_man_adj_commit(self, _event=None) -> None:
        self._save_man_adj_to_pickle()

    def _current_man_adj_value(self) -> float:
        try:
            return float(self.man_adj_var.get())
        except (tk.TclError, ValueError):
            return parse_man_adj(self._current_row().get("Man. Adj."))

    def _save_man_adj_to_pickle(self) -> None:
        if self._updating_man_adj_field:
            return
        try:
            value = float(self.man_adj_var.get())
        except (tk.TclError, ValueError):
            value = parse_man_adj(self._current_row().get("Man. Adj."))
            self._updating_man_adj_field = True
            self.man_adj_var.set(value)
            self._updating_man_adj_field = False
            return

        row_index = self._current_row_index()
        row = self.store["rows"][row_index]
        row["Man. Adj."] = value
        update_row_bc_corr_norm_trc(row, value)
        save_quant_store(self.app.quant_pickle_path, self.store)

    def _refresh_all(self) -> None:
        self._reset_event_modes()
        row = self._current_row()
        man_adj = parse_man_adj(row.get("Man. Adj."))
        self._updating_man_adj_field = True
        self.man_adj_var.set(man_adj)
        self._updating_man_adj_field = False
        self._sync_fit_controls_from_row(row)

        smooth, _baseline, _normalized = compute_row_mark_event_traces(row, man_adj)
        if smooth is not None and smooth.size:
            self._man_adj_step = max(1e-6, float(np.nanstd(smooth)) * 0.01)
        else:
            self._man_adj_step = 1.0

        row_index = self._current_row_index()
        self.row_label.config(
            text=f"Row {row_index + 1}  ({self.entry_pos + 1} / {len(self.entries)})"
        )
        self._draw_image()
        self._draw_traces()
        self.canvas.draw_idle()

    def _draw_image(self) -> None:
        app = self.app
        z_average = app.z_average
        assert z_average is not None
        height, width = z_average.shape
        area_map = app._ensure_area_map_for_mark_events()

        self.ax_image.clear()
        self.ax_image.imshow(
            z_average,
            cmap="gray",
            extent=[0, width, height, 0],
            aspect="equal",
        )
        if area_map is not None:
            self.ax_image.imshow(
                area_map,
                cmap="inferno",
                alpha=0.5,
                extent=[0, width, height, 0],
                aspect="equal",
            )

        active_row_index = self._current_row_index()
        for row_index, row in self.entries:
            vertices = row.get("ROI pixels")
            if vertices is None:
                continue
            verts = np.asarray(vertices, dtype=float)
            if verts.ndim != 2 or len(verts) < 3:
                continue
            is_active = row_index == active_row_index
            patch = Polygon(
                verts,
                closed=True,
                fill=False,
                edgecolor="lime" if is_active else "black",
                linewidth=2.2 if is_active else 0.9,
                zorder=4 if is_active else 3,
            )
            self.ax_image.add_patch(patch)

        self.ax_image.set_title("Z-average + area heatmap")
        self.ax_image.set_xlabel("x (px)")
        self.ax_image.set_ylabel("y (px)")

    def _draw_traces(self, man_adj: float | None = None) -> None:
        app = self.app
        row_index = self._current_row_index()
        row = self._current_row()
        if man_adj is None:
            man_adj = self._current_man_adj_value()
        smooth, baseline, normalized = compute_row_mark_event_traces(row, man_adj)

        self.ax_upper.clear()
        self.ax_lower.clear()

        xlabel = app._trace_xlabel()
        if smooth is None or baseline is None or normalized is None:
            message = "BC-corrected traces require ROI, BG, and bleach correction data."
            self.ax_upper.text(
                0.5,
                0.5,
                message,
                transform=self.ax_upper.transAxes,
                ha="center",
                va="center",
            )
            self.ax_upper.set_title(f"Row {row_index + 1}: smoothed BC-corrected")
            self.ax_lower.set_title(f"Row {row_index + 1}: normalized to BC baseline")
            self.ax_upper.set_xlabel(xlabel)
            self.ax_lower.set_xlabel(xlabel)
            self.canvas.draw_idle()
            return

        n_frames = len(smooth)
        x = app._frames_to_axis(np.arange(n_frames, dtype=float), one_based=False)
        frame_to_axis = lambda frame: app._frame_to_axis(frame, one_based=False)

        self.ax_upper.plot(x, smooth, color="0.15", linewidth=1.0, label="smoothed")
        self.ax_upper.plot(
            x,
            baseline,
            color="red",
            linestyle=":",
            linewidth=1.2,
            label="BC baseline",
        )
        draw_row_event_spans(self.ax_upper, row, n_frames, frame_to_axis)
        self.ax_upper.legend(loc="upper right", fontsize=8)
        self.ax_upper.set_ylabel("Intensity")
        self.ax_upper.set_title(f"Row {row_index + 1}: smoothed BC-corrected")
        self.ax_upper.set_xlabel(xlabel)

        self.ax_lower.plot(x, normalized, color="0.15", linewidth=1.0, zorder=3)
        self.ax_lower.axhline(1.0, color="red", linestyle=":", linewidth=1.2, label="baseline (=1)", zorder=4)
        draw_row_event_spans(self.ax_lower, row, n_frames, frame_to_axis)
        draw_locked_marked_events(self.ax_lower, row, n_frames, frame_to_axis)
        self._draw_draft_boundaries(n_frames, frame_to_axis)
        self.ax_lower.legend(loc="upper right", fontsize=8)
        self.ax_lower.set_ylabel("Normalized")
        self.ax_lower.set_title(f"Row {row_index + 1}: normalized to BC baseline")
        self.ax_lower.set_xlabel(xlabel)
        self.ax_upper.set_xlim(x[0], x[-1])
        self.ax_lower.set_xlim(x[0], x[-1])
        self.canvas.draw_idle()


class EditableROI:
    """Freehand polygon ROI: draw, move, vertex-adjust, delete."""

    def __init__(
        self,
        ax,
        image_shape: tuple[int, int],
        on_change,
        on_mode_change=None,
        edge_color: str = "lime",
        fill_color: str | None = None,
        fill_alpha: float = 0.0,
        handle_face_color: str = "yellow",
        preview_color: str | None = None,
        zorder: int = 5,
        sibling: "EditableROI | None" = None,
        enable_delete_key: bool = True,
        on_press_intercept: Callable | None = None,
    ):
        self.ax = ax
        self.height, self.width = image_shape
        self.on_change = on_change
        self.on_mode_change = on_mode_change
        self.on_press_intercept = on_press_intercept
        self.edge_color = edge_color
        self.fill_color = fill_color or edge_color
        self.fill_alpha = fill_alpha
        self.handle_face_color = handle_face_color
        self.preview_color = preview_color or edge_color
        self.zorder = zorder
        self.sibling = sibling
        self.enable_delete_key = enable_delete_key

        self.vertices: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.patch: Polygon | None = None
        self.handle_artists = None
        self.preview_line = None
        self.draw_mode = False

        self._press_origin: tuple[float, float] | None = None
        self._drag_mode: str | None = None
        self._anchor_vertices: np.ndarray | None = None
        self._current_stroke: list[tuple[float, float]] = []
        self._edit_vertex_idx: int | None = None

        self.ax.set_navigate(False)

        self.cid_press = ax.figure.canvas.mpl_connect("button_press_event", self._on_press)
        self.cid_release = ax.figure.canvas.mpl_connect("button_release_event", self._on_release)
        self.cid_motion = ax.figure.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.cid_key = ax.figure.canvas.mpl_connect("key_press_event", self._on_key)

    @property
    def roi(self) -> np.ndarray | None:
        return self.mask

    def set_draw_mode(self, enabled: bool) -> None:
        self.draw_mode = enabled
        self._update_handles()

    def _notify_mode_change(self) -> None:
        if self.on_mode_change is not None:
            self.on_mode_change()

    def clear(self) -> None:
        self.vertices = None
        self.mask = None
        self.draw_mode = False
        self._drag_mode = None
        self._edit_vertex_idx = None
        self._clear_preview()
        self._purge_display()
        self._update_handles()
        self.on_change()
        self.ax.figure.canvas.draw_idle()

    def _purge_display(self) -> None:
        """Remove ROI polygon/handles from the axes, including stale duplicates."""
        self._remove_artist(self.patch)
        self.patch = None
        self._remove_artist(self.handle_artists)
        self.handle_artists = None
        self._remove_artist(self.preview_line)
        self.preview_line = None

        target_edge = np.array(mcolors.to_rgba(self.edge_color))
        for artist in list(self.ax.patches):
            if not isinstance(artist, Polygon):
                continue
            if artist.get_zorder() != self.zorder:
                continue
            try:
                edge = np.asarray(artist.get_edgecolor())
                if edge.shape == target_edge.shape and np.allclose(edge, target_edge, atol=0.1):
                    artist.remove()
            except (ValueError, AttributeError, TypeError):
                continue

    def _remove_artist(self, artist) -> None:
        if artist is not None:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        if artist is self.patch:
            self.patch = None
        if artist is self.handle_artists:
            self.handle_artists = None
        if artist is self.preview_line:
            self.preview_line = None

    def _artist_attached(self, artist) -> bool:
        if artist is None:
            return False
        axes = getattr(artist, "axes", None)
        return axes is not None

    def _clip_point(self, x: float, y: float) -> tuple[float, float]:
        return (
            float(np.clip(x, 0, self.width - 1)),
            float(np.clip(y, 0, self.height - 1)),
        )

    def _event_xy(self, event) -> tuple[float, float] | None:
        if event.xdata is not None and event.ydata is not None:
            return float(event.xdata), float(event.ydata)
        if event.x is None or event.y is None:
            return None
        return self.ax.transData.inverted().transform((event.x, event.y))

    def _contains_point(self, x: float, y: float) -> bool:
        if self.vertices is None or len(self.vertices) < 3:
            return False
        return bool(MplPath(self.vertices).contains_point((x, y), radius=1.0))

    def _nearest_vertex(self, x: float, y: float) -> int | None:
        if self.vertices is None:
            return None
        dists = np.hypot(self.vertices[:, 0] - x, self.vertices[:, 1] - y)
        idx = int(np.argmin(dists))
        if dists[idx] <= HANDLE_RADIUS:
            return idx
        return None

    def _simplify_vertices(self, vertices: np.ndarray) -> np.ndarray:
        if len(vertices) <= MAX_EDIT_VERTICES:
            return vertices
        step = max(1, len(vertices) // MAX_EDIT_VERTICES)
        return vertices[::step]

    def _finalize_stroke(self, stroke: list[tuple[float, float]]) -> None:
        if len(stroke) < MIN_FREEHAND_POINTS:
            return

        vertices = np.asarray(stroke, dtype=float)
        vertices = self._simplify_vertices(vertices)
        vertices = clip_vertices(vertices, self.width, self.height)

        self.vertices = vertices
        self.mask = polygon_mask(vertices, self.height, self.width)
        self._update_patch()
        self.draw_mode = False
        self._notify_mode_change()

    def _update_patch(self) -> None:
        if self.vertices is None or len(self.vertices) < 3:
            self._purge_display()
            self._update_handles()
            return

        if self.patch is not None and not self._artist_attached(self.patch):
            self.patch = None

        fill = self.fill_alpha > 0
        facecolor = mcolors.to_rgba(self.fill_color, self.fill_alpha) if fill else "none"
        if self.patch is None:
            self.patch = Polygon(
                self.vertices,
                closed=True,
                fill=fill,
                facecolor=facecolor,
                edgecolor=self.edge_color,
                linewidth=2,
                zorder=self.zorder,
            )
            self.ax.add_patch(self.patch)
        else:
            self.patch.set_xy(self.vertices)
        self._update_handles()

    def _update_handles(self) -> None:
        if self.handle_artists is not None and not self._artist_attached(self.handle_artists):
            self.handle_artists = None

        self._remove_artist(self.handle_artists)
        self.handle_artists = None

        if self.vertices is None or self.draw_mode or self._drag_mode == "draw":
            return

        (self.handle_artists,) = self.ax.plot(
            self.vertices[:, 0],
            self.vertices[:, 1],
            linestyle="none",
            marker="o",
            markersize=5,
            markerfacecolor=self.handle_face_color,
            markeredgecolor=self.edge_color,
            markeredgewidth=1,
            zorder=self.zorder + 1,
        )

    def _update_preview(self) -> None:
        if not self._current_stroke:
            return

        xs, ys = zip(*self._current_stroke)
        if self.preview_line is None:
            (self.preview_line,) = self.ax.plot(
                xs, ys, color=self.preview_color, linewidth=2, alpha=0.9, zorder=self.zorder + 1
            )
        else:
            self.preview_line.set_data(xs, ys)

    def _clear_preview(self) -> None:
        self._remove_artist(self.preview_line)
        self.preview_line = None
        self._current_stroke = []

    def _start_draw(self, x: float, y: float) -> None:
        self._clear_preview()
        self._drag_mode = "draw"
        x, y = self._clip_point(x, y)
        self._press_origin = (x, y)
        self._current_stroke = [(x, y)]
        self._update_preview()

    def _append_stroke_point(self, x: float, y: float) -> None:
        x, y = self._clip_point(x, y)
        if self._current_stroke:
            lx, ly = self._current_stroke[-1]
            if np.hypot(x - lx, y - ly) < MIN_POINT_SPACING:
                return
        self._current_stroke.append((x, y))
        self._update_preview()

    def _sibling_takes_priority(self, x: float, y: float) -> bool:
        sibling = self.sibling
        if sibling is None:
            return False
        if sibling.draw_mode or sibling._drag_mode is not None:
            return True
        if sibling.vertices is None:
            return False
        if sibling._nearest_vertex(x, y) is not None:
            return True
        if self.zorder < sibling.zorder and sibling._contains_point(x, y):
            return True
        return False

    def _on_press(self, event) -> None:
        if event.button != 1:
            return

        if self.on_press_intercept is not None and self.on_press_intercept(event):
            return

        xy = self._event_xy(event) if event.inaxes is self.ax else None
        if xy is None:
            return
        x, y = xy

        if self._sibling_takes_priority(x, y):
            return

        if self.vertices is not None:
            vertex_idx = self._nearest_vertex(x, y)
            if vertex_idx is not None:
                self._drag_mode = "edit_vertex"
                self._edit_vertex_idx = vertex_idx
                self._press_origin = (x, y)
                self._update_handles()
                self.ax.figure.canvas.draw_idle()
                return

            if self._contains_point(x, y):
                self._drag_mode = "move"
                self._press_origin = (x, y)
                self._anchor_vertices = self.vertices.copy()
                self.ax.figure.canvas.draw_idle()
                return

        if self.draw_mode or self.vertices is None:
            self._start_draw(x, y)
            self.ax.figure.canvas.draw_idle()

    def _on_motion(self, event) -> None:
        if self._drag_mode is None:
            return

        xy = self._event_xy(event)
        if xy is None:
            return
        x, y = xy

        if self._drag_mode == "draw":
            if event.inaxes is self.ax:
                self._append_stroke_point(x, y)
        elif self._drag_mode == "move" and self._anchor_vertices is not None and self._press_origin is not None:
            dx = x - self._press_origin[0]
            dy = y - self._press_origin[1]
            self.vertices = clip_vertices(self._anchor_vertices + np.array([dx, dy]), self.width, self.height)
            self._update_patch()
        elif self._drag_mode == "edit_vertex" and self._edit_vertex_idx is not None and self.vertices is not None:
            x, y = self._clip_point(x, y)
            self.vertices[self._edit_vertex_idx] = (x, y)
            self._update_patch()

        self.ax.figure.canvas.draw_idle()

    def _on_release(self, event) -> None:
        if self._drag_mode is None:
            return

        if self._drag_mode == "draw":
            xy = self._event_xy(event)
            if xy is not None and event.inaxes is self.ax:
                self._append_stroke_point(xy[0], xy[1])
            stroke = self._current_stroke
            self._clear_preview()
            if len(stroke) >= MIN_FREEHAND_POINTS:
                self._finalize_stroke(stroke)
                self.on_change()
        elif self._drag_mode in ("move", "edit_vertex") and self.vertices is not None:
            self.mask = polygon_mask(self.vertices, self.height, self.width)
            self.on_change()

        self._drag_mode = None
        self._press_origin = None
        self._anchor_vertices = None
        self._edit_vertex_idx = None
        self._update_handles()
        self.ax.figure.canvas.draw_idle()

    def _on_key(self, event) -> None:
        if not self.enable_delete_key:
            return
        if event.key in ("delete", "backspace"):
            self.clear()
            self.ax.figure.canvas.draw_idle()


class StackAnalyzerApp:
    def __init__(self, initial_path: str | None = None):
        self.stack: np.ndarray | None = None
        self.z_average: np.ndarray | None = None
        self.raw_trace: np.ndarray | None = None
        self.raw_bg_trace: np.ndarray | None = None
        self.smooth_trace: np.ndarray | None = None
        self.bc_baseline_trace: np.ndarray | None = None
        self.file_path = initial_path or ""

        self.n_frames = 0
        self.extension = 50
        self.baseline_fraction = 0.2
        self.start_frames = parse_start_frames(DEFAULT_STARTS, 0)
        self.starts_text = DEFAULT_STARTS
        self.acq_fps = DEFAULT_ACQ_FPS
        self.avr_factor = float(DEFAULT_AVR)
        self.convert_time_axis = False
        self.baseline_level = 1.0
        self.rel_x: np.ndarray | None = None
        self.normalized_segments: list[np.ndarray] = []
        self.mean_trace_values: np.ndarray | None = None
        self.segment_baseline_len = 1
        self.computed_area = 0.0
        self.heatmap_enabled = False
        self.pixel_mean_trace: np.ndarray | None = None
        self.pixel_rel_x: np.ndarray | None = None
        self.base_image = None
        self.heatmap_overlay = None
        self.heatmap_colorbar = None
        self.area_map_cache: np.ndarray | None = None
        self._heatmap_progress_artists: list = []
        self._heatmap_traces_dirty = True
        self._heatmap_busy = False
        self._heatmap_pending = False
        self._heatmap_pending_full = False
        self._block_area_slider_callbacks = False
        self.quant_pickle_path: Path | None = None
        self.show_saved_rois = False
        self._saved_roi_overlays: list[tuple[Polygon, int]] = []
        self._active_saved_roi_row_index: int | None = None
        self._loading_saved_roi = False
        self._applying_quant_settings = False
        self._suppress_bg_change_prompt = False
        self._mark_events_window: MarkEventsWindow | None = None

        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title("Stack Analyzer")

        gs = self.fig.add_gridspec(
            4,
            3,
            height_ratios=[0.08, 0.31, 0.31, 0.30],
            width_ratios=[1, 1, 1],
            hspace=0.35,
            wspace=0.12,
        )
        gs.update(top=0.82, bottom=0.06, left=0.05, right=0.98)

        left_gs = gs[1:4, 0].subgridspec(3, 1, height_ratios=[0.86, 0.07, 0.07], hspace=0.06)
        image_row_gs = left_gs[0, 0].subgridspec(1, 2, width_ratios=[1, 0.06], wspace=0.05)
        self.ax_image = self.fig.add_subplot(image_row_gs[0, 0])
        self.ax_heatmap_cbar = self.fig.add_subplot(image_row_gs[0, 1])
        self.ax_heatmap_cbar.set_axis_off()
        show_rois_ax = self.fig.add_subplot(left_gs[1, 0])
        show_rois_ax.set_axis_off()
        self.check_show_rois = widgets.CheckButtons(show_rois_ax, ["show ROIs"], [False])
        self.check_show_rois.on_clicked(self._on_show_rois_toggled)
        toggle_ax = self.fig.add_subplot(left_gs[2, 0])
        toggle_ax.set_axis_off()
        self.check_heatmap = widgets.CheckButtons(toggle_ax, ["Heatmap"], [False])
        self.check_heatmap.on_clicked(self._on_heatmap_toggled)

        self.ax_raw = self.fig.add_subplot(gs[1, 1:3])
        self.ax_smooth = self.fig.add_subplot(gs[2, 1:3], sharex=self.ax_raw)
        self.ax_segments = self.fig.add_subplot(gs[3, 1:3])

        self._build_controls()
        self.roi_tool: EditableROI | None = None
        self.bg_roi_tool: EditableROI | None = None

        if initial_path:
            self.load_stack(initial_path)

    def _build_controls(self) -> None:
        self.file_text = self.fig.text(
            0.05, 0.975, "No file loaded", fontsize=9, va="top", ha="left"
        )

        ax_browse = self.fig.add_axes([0.05, 0.905, 0.08, 0.035])
        self.btn_browse = widgets.Button(ax_browse, "Browse…")
        self.btn_browse.on_clicked(self._browse_file)

        ax_inspect_pickle = self.fig.add_axes([0.14, 0.905, 0.11, 0.035])
        self.btn_inspect_pickle = widgets.Button(ax_inspect_pickle, "Inspect Pickle")
        self.btn_inspect_pickle.on_clicked(self._on_inspect_pickle)

        ax_draw = self.fig.add_axes([0.05, 0.855, 0.08, 0.035])
        self.btn_draw = widgets.Button(ax_draw, "Draw ROI")
        self.btn_draw.on_clicked(self._toggle_draw_mode)

        ax_clear = self.fig.add_axes([0.14, 0.855, 0.08, 0.035])
        self.btn_clear = widgets.Button(ax_clear, "Clear ROI")
        self.btn_clear.on_clicked(lambda _event: self._clear_roi())

        ax_draw_bg = self.fig.add_axes([0.05, 0.815, 0.08, 0.035])
        self.btn_draw_bg = widgets.Button(ax_draw_bg, "Draw BG")
        self.btn_draw_bg.on_clicked(self._toggle_bg_draw_mode)

        ax_clear_bg = self.fig.add_axes([0.14, 0.815, 0.08, 0.035])
        self.btn_clear_bg = widgets.Button(ax_clear_bg, "Clear BG")
        self.btn_clear_bg.on_clicked(lambda _event: self._clear_bg_roi())

        ax_save_roi = self.fig.add_axes([0.05, 0.775, 0.08, 0.035])
        self.btn_save_roi = widgets.Button(ax_save_roi, "Save ROI")
        self.btn_save_roi.on_clicked(self._on_save_roi)

        ax_delete_roi = self.fig.add_axes([0.14, 0.775, 0.08, 0.035])
        self.btn_delete_roi = widgets.Button(ax_delete_roi, "Delete ROI")
        self.btn_delete_roi.on_clicked(self._on_delete_roi)

        ax_mark_events = self.fig.add_axes([0.23, 0.775, 0.10, 0.035])
        self.btn_mark_events = widgets.Button(ax_mark_events, "Mark Events")
        self.btn_mark_events.on_clicked(self._on_mark_events)

        ax_window = self.fig.add_axes([0.30, 0.905, 0.18, 0.025])
        self.slider_window = widgets.Slider(ax_window, "SG window", 3, 501, valinit=51, valstep=2)
        self.slider_window.on_changed(lambda _val: self._update_analysis())

        ax_poly = self.fig.add_axes([0.30, 0.855, 0.18, 0.025])
        self.slider_poly = widgets.Slider(ax_poly, "SG order", 1, 7, valinit=3, valstep=1)
        self.slider_poly.on_changed(lambda _val: self._update_analysis())

        ax_ext = self.fig.add_axes([0.52, 0.905, 0.18, 0.025])
        self.slider_extension = widgets.Slider(ax_ext, "Extension", 5, 500, valinit=50, valstep=1)
        self.slider_extension.on_changed(lambda _val: self._update_analysis())

        ax_starts = self.fig.add_axes([0.52, 0.855, 0.18, 0.035])
        self.text_starts = widgets.TextBox(ax_starts, "Starts", initial=self.starts_text)
        self.text_starts.on_submit(self._on_starts_changed)
        self._patch_textbox_resize(self.text_starts)

        ax_freq = self.fig.add_axes([0.52, 0.800, 0.11, 0.032])
        self.text_freq = widgets.TextBox(ax_freq, "Freq (fps)", initial=str(DEFAULT_ACQ_FPS))
        self.text_freq.on_submit(self._on_timing_changed)
        self._patch_textbox_resize(self.text_freq)

        ax_avr = self.fig.add_axes([0.64, 0.800, 0.05, 0.032])
        self.text_avr = widgets.TextBox(ax_avr, "Avr", initial=str(DEFAULT_AVR))
        self.text_avr.on_submit(self._on_timing_changed)
        self._patch_textbox_resize(self.text_avr)

        self.effective_fps_text = self.fig.text(
            0.70, 0.816, "", fontsize=9, va="center", ha="left"
        )
        self._update_effective_fps_display()

        ax_convert = self.fig.add_axes([0.78, 0.798, 0.07, 0.034])
        ax_convert.set_axis_off()
        self.check_convert = widgets.CheckButtons(ax_convert, ["convert"], [False])
        self.check_convert.on_clicked(self._on_convert_toggled)

        ax_area_left = self.fig.add_axes([0.74, 0.905, 0.18, 0.025])
        self.slider_area_left = widgets.Slider(ax_area_left, "Area L", 1, 60, valinit=1, valstep=1)
        self.slider_area_left.on_changed(lambda _val: self._on_area_slider_changed())

        ax_area_right = self.fig.add_axes([0.74, 0.855, 0.18, 0.025])
        self.slider_area_right = widgets.Slider(ax_area_right, "Area R", 1, 60, valinit=60, valstep=1)
        self.slider_area_right.on_changed(lambda _val: self._on_area_slider_changed())

    @staticmethod
    def _patch_textbox_resize(textbox: widgets.TextBox) -> None:
        """Work around matplotlib 3.11 passing ResizeEvent through a mouse-event wrapper."""
        if not textbox._cids:
            return
        resize_cid = textbox._cids.pop()
        textbox.canvas.mpl_disconnect(resize_cid)

        def on_resize(event) -> None:
            textbox.stop_typing()

        textbox.connect_event("resize_event", on_resize)

    def _browse_file(self, _event) -> None:
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select TIFF stack",
            filetypes=[("TIFF files", "*.tif *.tiff *.TIFF *.TIF"), ("All files", "*.*")],
        )
        root.destroy()
        if path:
            self.load_stack(path)

    def _on_inspect_pickle(self, _event) -> None:
        path = self.quant_pickle_path
        if path is None:
            root = tk.Tk()
            root.withdraw()
            path_str = filedialog.askopenfilename(
                title="Select ROI quantification pickle",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            )
            root.destroy()
            if not path_str:
                return
            path = Path(path_str)
        elif not path.exists():
            messagebox.showinfo(
                "Inspect Pickle",
                f"No pickle file found at:\n{path}\n\nLoad a TIFF stack to create one.",
            )
            return
        open_quant_pickle_inspector(
            path,
            current_directory=(
                os.path.dirname(os.path.abspath(self.file_path))
                if self.file_path
                else None
            ),
        )

    def _toggle_draw_mode(self, _event) -> None:
        if self.roi_tool is None:
            return
        if self.bg_roi_tool is not None and self.bg_roi_tool.draw_mode:
            self.bg_roi_tool.set_draw_mode(False)
            self._sync_bg_draw_button()
        enabling = not self.roi_tool.draw_mode
        if enabling:
            self._active_saved_roi_row_index = None
            self.roi_tool.vertices = None
            self.roi_tool.mask = None
            self.roi_tool._purge_display()
            self.roi_tool._update_handles()
            if self.show_saved_rois:
                self._update_saved_roi_display()
        self.roi_tool.set_draw_mode(not self.roi_tool.draw_mode)
        self._sync_draw_button()

    def _toggle_bg_draw_mode(self, _event) -> None:
        if self.bg_roi_tool is None:
            return
        if self.roi_tool is not None and self.roi_tool.draw_mode:
            self.roi_tool.set_draw_mode(False)
            self._sync_draw_button()
        self.bg_roi_tool.set_draw_mode(not self.bg_roi_tool.draw_mode)
        self._sync_bg_draw_button()

    def _on_heatmap_toggled(self, _label: str) -> None:
        self.heatmap_enabled = bool(self.check_heatmap.get_status()[0])
        self._update_heatmap_display()

    def _on_show_rois_toggled(self, _label: str) -> None:
        self.show_saved_rois = bool(self.check_show_rois.get_status()[0])
        if not self.show_saved_rois:
            self._deselect_saved_roi(clear_traces=False)
        self._update_saved_roi_display()

    def _clear_saved_roi_overlays(self) -> None:
        for patch, _row_idx in self._saved_roi_overlays:
            try:
                patch.remove()
            except (ValueError, AttributeError):
                pass
        self._saved_roi_overlays = []

    def _stack_row_entries(self, store: dict | None = None) -> list[tuple[int, dict]]:
        if self.stack is None or not self.file_path:
            return []
        if store is None:
            if self.quant_pickle_path is None:
                return []
            store = load_quant_store(self.quant_pickle_path)

        current_dir = os.path.dirname(os.path.abspath(self.file_path))
        current_size = format_stack_size(
            self.stack.shape[2], self.stack.shape[1], self.stack.shape[0]
        )
        entries: list[tuple[int, dict]] = []
        for row_index, row in enumerate(store.get("rows", [])):
            if not directory_matches(row.get("directory"), current_dir):
                continue
            if row.get("size") != current_size:
                continue
            entries.append((row_index, row))
        return entries

    def _first_stack_row(self, store: dict) -> dict | None:
        entries = self._stack_row_entries(store)
        return entries[0][1] if entries else None

    def _first_stack_row_with_bg(self, store: dict) -> dict | None:
        for _row_index, row in self._stack_row_entries(store):
            if row.get("BG pixels") is not None:
                return row
        return None

    def _get_stack_bg_from_store(
        self, store: dict
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return stack-canonical BG pixels and trace from the pickle (first row that has BG)."""
        for _row_index, row in self._stack_row_entries(store):
            pixels = row.get("BG pixels")
            if pixels is None:
                continue
            pixels_arr = np.asarray(pixels, dtype=np.float64).copy()
            trc = row.get("BG trc")
            if trc is not None:
                trc_arr = np.asarray(trc, dtype=np.float64).copy()
            else:
                trc_arr = self._compute_bg_trace_from_pixels(pixels_arr)
            return pixels_arr, trc_arr
        return None, None

    def _resolve_bg_fields_for_quant_row(
        self, store: dict
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Preserve stack BG from the pickle; fall back to the GUI only when no row has BG yet."""
        canonical_pixels, canonical_trc = self._get_stack_bg_from_store(store)
        if canonical_pixels is not None:
            return canonical_pixels, canonical_trc

        ui_pixels = None
        if self.bg_roi_tool is not None and self.bg_roi_tool.vertices is not None:
            ui_pixels = np.asarray(self.bg_roi_tool.vertices, dtype=np.float64).copy()
        ui_trc = None
        if self.raw_bg_trace is not None:
            ui_trc = np.asarray(self.raw_bg_trace, dtype=np.float64).copy()
        elif ui_pixels is not None:
            ui_trc = self._compute_bg_trace_from_pixels(ui_pixels)
        return ui_pixels, ui_trc

    def _load_stack_bg_into_ui(self, store: dict) -> None:
        bg_row = self._first_stack_row_with_bg(store)
        if bg_row is not None:
            self._load_bg_from_row(bg_row)
        elif self.bg_roi_tool is not None:
            self.bg_roi_tool.clear()
            self.raw_bg_trace = None

    def _stack_rows_have_mismatched_bg(self, store: dict) -> bool:
        entries = self._stack_row_entries(store)
        if len(entries) <= 1:
            return False

        reference_pixels = None
        reference_defined = False
        for _row_index, row in entries:
            pixels = row.get("BG pixels")
            if pixels is None:
                if reference_defined:
                    return True
                continue
            if not reference_defined:
                reference_pixels = pixels
                reference_defined = True
            elif not bg_pixels_equal(reference_pixels, pixels):
                return True
        return False

    def _compute_bg_trace_from_pixels(self, bg_pixels) -> np.ndarray | None:
        if self.stack is None or self.bg_roi_tool is None or bg_pixels is None:
            return None
        verts = clip_vertices(
            np.asarray(bg_pixels, dtype=np.float64),
            self.bg_roi_tool.width,
            self.bg_roi_tool.height,
        )
        mask = polygon_mask(verts, self.bg_roi_tool.height, self.bg_roi_tool.width)
        return compute_raw_trace(self.stack, mask)

    def _copy_bg_from_first_stack_row(self, store: dict) -> bool:
        bg_row = self._first_stack_row_with_bg(store)
        if bg_row is None:
            return False

        bg_pixels = bg_row.get("BG pixels")
        bg_trace = self._compute_bg_trace_from_pixels(bg_pixels)
        if bg_pixels is not None and bg_trace is None:
            bg_trace = bg_row.get("BG trc")

        for _row_index, row in self._stack_row_entries(store):
            if bg_pixels is None:
                row["BG pixels"] = None
                row["BG trc"] = None
                row["bleach correct"] = None
                row["BC baseline"] = None
                row[FIT_PARAMS_COLUMN] = None
                row[BC_CORR_NORM_TRC_COLUMN] = None
            else:
                row["BG pixels"] = np.asarray(bg_pixels, dtype=np.float64).copy()
                row["BG trc"] = (
                    None
                    if bg_trace is None
                    else np.asarray(bg_trace, dtype=np.float64).copy()
                )
                if row["BG trc"] is not None:
                    update_row_bleach_correction(row, force_auto_fit=True)
                    update_row_bc_corr_norm_trc(row)
        return True

    def _ensure_unified_bg_on_load(self, store: dict) -> bool:
        if not self._stack_rows_have_mismatched_bg(store):
            return False
        if not messagebox.askyesno(
            "BG ROI mismatch",
            "This pickle file has different BG ROIs across rows for the loaded stack.\n\n"
            "Copy the BG ROI from row 1 to all other rows for this stack?",
        ):
            return False
        return self._copy_bg_from_first_stack_row(store)

    def _ask_apply_new_bg_roi(self) -> bool:
        return messagebox.askyesno(
            "BG ROI changed",
            "A new BG ROI was drawn.\n\n"
            "Yes: replace the BG ROI in all rows for this stack.\n"
            "No: discard this change and restore the previous BG ROI.",
        )

    def _apply_bg_to_all_stack_rows(
        self,
        bg_pixels: np.ndarray | None,
        bg_trace: np.ndarray | None,
    ) -> None:
        if self.quant_pickle_path is None:
            return

        store = load_quant_store(self.quant_pickle_path)
        pixels_copy = None if bg_pixels is None else np.asarray(bg_pixels, dtype=np.float64).copy()
        trace_copy = None if bg_trace is None else np.asarray(bg_trace, dtype=np.float64).copy()

        for _row_index, row in self._stack_row_entries(store):
            row["BG pixels"] = None if pixels_copy is None else pixels_copy.copy()
            row["BG trc"] = None if trace_copy is None else trace_copy.copy()
            if trace_copy is None:
                row["bleach correct"] = None
                row["BC baseline"] = None
                row[FIT_PARAMS_COLUMN] = None
                row[BC_CORR_NORM_TRC_COLUMN] = None
            else:
                update_row_bleach_correction(row, force_auto_fit=True)
                update_row_bc_corr_norm_trc(row)

        save_quant_store(self.quant_pickle_path, store)

    def _restore_canonical_bg_roi(self) -> None:
        if self.quant_pickle_path is None:
            self.raw_bg_trace = None
            if self.bg_roi_tool is not None:
                self._suppress_bg_change_prompt = True
                try:
                    self.bg_roi_tool.clear()
                finally:
                    self._suppress_bg_change_prompt = False
            return

        store = load_quant_store(self.quant_pickle_path)
        self._suppress_bg_change_prompt = True
        try:
            self._load_stack_bg_into_ui(store)
        finally:
            self._suppress_bg_change_prompt = False

    def _saved_roi_entries_for_current_stack(self) -> list[tuple[int, dict]]:
        if self.quant_pickle_path is None:
            return []
        store = load_quant_store(self.quant_pickle_path)
        return self._stack_row_entries(store)

    def _hit_test_saved_roi(self, x: float, y: float) -> int | None:
        for _patch, row_index in reversed(self._saved_roi_overlays):
            if row_index == self._active_saved_roi_row_index:
                continue
            store = load_quant_store(self.quant_pickle_path)
            row = store["rows"][row_index]
            vertices = row.get("ROI pixels")
            if vertices is None:
                continue
            verts = np.asarray(vertices, dtype=float)
            if verts.ndim != 2 or len(verts) < 3:
                continue
            if MplPath(verts).contains_point((x, y), radius=2.0):
                return row_index
        return None

    def _saved_roi_press_intercept(self, event) -> bool:
        if not self.show_saved_rois or event.inaxes is not self.ax_image:
            return False
        if event.xdata is None or event.ydata is None:
            return False

        x, y = float(event.xdata), float(event.ydata)
        row_index = self._hit_test_saved_roi(x, y)
        if row_index is not None:
            self._activate_saved_roi(row_index)
            return True

        if self.roi_tool is not None and self.roi_tool.vertices is not None:
            if self.roi_tool._contains_point(x, y):
                return False
            if self.roi_tool._nearest_vertex(x, y) is not None:
                return False

        if self._active_saved_roi_row_index is not None or (
            self.roi_tool is not None and self.roi_tool.vertices is not None
        ):
            if not (self.roi_tool is not None and self.roi_tool.draw_mode):
                self._deselect_saved_roi()
                return True
        return False

    def _current_quant_settings(self) -> dict:
        f_left = int(self.slider_area_left.val)
        f_right = int(self.slider_area_right.val)
        if f_right < f_left:
            f_left, f_right = f_right, f_left
        return {
            "starts": format_start_frames(self.start_frames),
            "acq_fps": self.acq_fps,
            "avr_factor": self.avr_factor,
            "sg_window": int(self.slider_window.val),
            "sg_poly": int(self.slider_poly.val),
            "extension": int(self.slider_extension.val),
            "area_left": f_left,
            "area_right": f_right,
        }

    def _ask_quant_settings_choice(self, pickle_settings: dict, current_settings: dict) -> str | None:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        message = (
            "The analysis settings in the pickle file differ from the current GUI.\n\n"
            f"Pickle: starts={pickle_settings['starts']}, "
            f"{format_freq_avr_field(pickle_settings['acq_fps'], pickle_settings['avr_factor'])}, "
            f"SG={format_sg_field(pickle_settings['sg_window'], pickle_settings['sg_poly'])}, "
            f"extension={pickle_settings['extension']}, "
            f"Area L+R={format_area_lr_field(pickle_settings['area_left'], pickle_settings['area_right'])}\n\n"
            f"Current: starts={current_settings['starts']}, "
            f"{format_freq_avr_field(current_settings['acq_fps'], current_settings['avr_factor'])}, "
            f"SG={format_sg_field(current_settings['sg_window'], current_settings['sg_poly'])}, "
            f"extension={current_settings['extension']}, "
            f"Area L+R={format_area_lr_field(current_settings['area_left'], current_settings['area_right'])}\n\n"
            "Which settings should be used for all rows in the pickle file?"
        )

        choice: dict[str, str | None] = {"value": None}

        dialog = tk.Toplevel(root)
        dialog.title("Analysis settings differ")
        dialog.attributes("-topmost", True)
        dialog.grab_set()
        tk.Label(dialog, text=message, justify="left", wraplength=520, padx=12, pady=12).pack()
        button_row = tk.Frame(dialog, padx=12, pady=8)
        button_row.pack()

        def choose(value: str) -> None:
            choice["value"] = value
            dialog.destroy()
            root.destroy()

        tk.Button(
            button_row,
            text="Use pickle settings",
            width=20,
            command=lambda: choose("pickle"),
        ).pack(side="left", padx=4)
        tk.Button(
            button_row,
            text="Use current GUI settings",
            width=20,
            command=lambda: choose("current"),
        ).pack(side="left", padx=4)
        tk.Button(button_row, text="Cancel", width=10, command=lambda: choose(None)).pack(
            side="left", padx=4
        )
        dialog.protocol("WM_DELETE_WINDOW", lambda: choose(None))
        root.wait_window(dialog)
        return choice["value"]

    def _pickle_settings_for_stack(self, store: dict) -> dict | None:
        if self.stack is None or not self.file_path or not store.get("rows"):
            return None

        current_dir = os.path.dirname(os.path.abspath(self.file_path))
        current_size = format_stack_size(
            self.stack.shape[2], self.stack.shape[1], self.n_frames
        )
        for row in store["rows"]:
            if not directory_matches(row.get("directory"), current_dir):
                continue
            if row.get("size") != current_size:
                continue
            try:
                return quant_settings_from_row(row)
            except (ValueError, KeyError, TypeError):
                continue
        return None

    def _apply_quant_settings_to_gui(self, settings: dict) -> None:
        self._applying_quant_settings = True
        self.acq_fps = settings["acq_fps"]
        self.avr_factor = settings["avr_factor"]
        self.text_freq.set_val(str(settings["acq_fps"]))
        self.text_avr.set_val(
            str(int(settings["avr_factor"]))
            if float(settings["avr_factor"]).is_integer()
            else str(settings["avr_factor"])
        )
        self._update_effective_fps_display()
        sg_window = min(int(settings["sg_window"]), int(self.slider_window.valmax))
        sg_window = max(3, sg_window if sg_window % 2 else sg_window - 1)
        self.slider_window.set_val(sg_window)
        self.slider_poly.set_val(settings["sg_poly"])
        self.slider_extension.set_val(settings["extension"])
        self.start_frames = parse_start_frames(settings["starts"], self.n_frames)
        self._sync_starts_textbox_from_frames(self.start_frames)
        self._block_area_slider_callbacks = True
        try:
            self.slider_area_left.set_val(settings["area_left"])
            self.slider_area_right.set_val(settings["area_right"])
        finally:
            self._block_area_slider_callbacks = False
        self._applying_quant_settings = False
        self._update_analysis()

    def _apply_quant_settings_to_all_rows(self, settings: dict) -> None:
        if self.quant_pickle_path is None:
            return
        store = load_quant_store(self.quant_pickle_path)
        fields = quant_settings_to_row_fields(settings)
        for row in store["rows"]:
            row.update(fields)
        save_quant_store(self.quant_pickle_path, store)

    def _resolve_quant_settings_conflict(self, row: dict) -> bool:
        if self._applying_quant_settings:
            return True
        pickle_settings = quant_settings_from_row(row)
        current_settings = self._current_quant_settings()
        if quant_settings_equal(pickle_settings, current_settings):
            return True

        choice = self._ask_quant_settings_choice(pickle_settings, current_settings)
        if choice is None:
            return False
        chosen = pickle_settings if choice == "pickle" else current_settings
        self._apply_quant_settings_to_all_rows(chosen)
        self._apply_quant_settings_to_gui(chosen)
        return True

    def _load_vertices_into_roi_tool(self, vertices: np.ndarray) -> None:
        if self.roi_tool is None:
            return
        verts = clip_vertices(np.asarray(vertices, dtype=float), self.roi_tool.width, self.roi_tool.height)
        self.roi_tool.vertices = verts
        self.roi_tool.mask = polygon_mask(verts, self.roi_tool.height, self.roi_tool.width)
        self.roi_tool.draw_mode = False
        self.roi_tool._notify_mode_change()
        self.roi_tool._update_patch()

    def _load_bg_from_row(self, row: dict, *, suppress_prompt: bool = True) -> None:
        if self.bg_roi_tool is None:
            return
        previous = self._suppress_bg_change_prompt
        if suppress_prompt:
            self._suppress_bg_change_prompt = True
        try:
            bg_vertices = row.get("BG pixels")
            if bg_vertices is None:
                self.bg_roi_tool.clear()
                self.raw_bg_trace = None
                return
            verts = clip_vertices(
                np.asarray(bg_vertices, dtype=float),
                self.bg_roi_tool.width,
                self.bg_roi_tool.height,
            )
            self.bg_roi_tool.vertices = verts
            self.bg_roi_tool.mask = polygon_mask(verts, self.bg_roi_tool.height, self.bg_roi_tool.width)
            self.bg_roi_tool.draw_mode = False
            self.bg_roi_tool._notify_mode_change()
            self.bg_roi_tool._update_patch()
            if self.stack is not None:
                self.raw_bg_trace = compute_raw_trace(self.stack, self.bg_roi_tool.mask)
        finally:
            if suppress_prompt:
                self._suppress_bg_change_prompt = previous

    def _activate_saved_roi(self, row_index: int) -> None:
        if self.quant_pickle_path is None or self.stack is None or self.roi_tool is None:
            return

        store = load_quant_store(self.quant_pickle_path)
        if row_index < 0 or row_index >= len(store["rows"]):
            return
        row = store["rows"][row_index]
        if not self._resolve_quant_settings_conflict(row):
            return

        self._loading_saved_roi = True
        try:
            self._active_saved_roi_row_index = row_index
            self._load_vertices_into_roi_tool(np.asarray(row["ROI pixels"], dtype=float))
            self._load_bg_from_row(row)
            self._on_roi_changed()
            self._update_saved_roi_display()
        finally:
            self._loading_saved_roi = False

    def _deselect_saved_roi(self, *, clear_traces: bool = True) -> None:
        self._active_saved_roi_row_index = None
        if self.roi_tool is not None:
            self.roi_tool.set_draw_mode(False)
            self._sync_draw_button()
            self.roi_tool.vertices = None
            self.roi_tool.mask = None
            self.roi_tool._purge_display()
            self.roi_tool._update_handles()
        if clear_traces:
            self.raw_trace = None
            self._update_roi_traces()

    def _sync_active_quant_row(self) -> None:
        if (
            self._active_saved_roi_row_index is None
            or self.quant_pickle_path is None
            or self.roi_tool is None
            or self.roi_tool.vertices is None
        ):
            return
        store = load_quant_store(self.quant_pickle_path)
        row_index = self._active_saved_roi_row_index
        if row_index < 0 or row_index >= len(store["rows"]):
            return
        store["rows"][row_index] = self._build_quant_row(
            existing_row=store["rows"][row_index],
            store=store,
        )
        save_quant_store(self.quant_pickle_path, store)

    def _update_saved_roi_display(self) -> None:
        self._clear_saved_roi_overlays()
        if not self.show_saved_rois or self.stack is None:
            self.fig.canvas.draw_idle()
            return

        for display_idx, (row_index, row) in enumerate(self._saved_roi_entries_for_current_stack()):
            if row_index == self._active_saved_roi_row_index:
                continue
            vertices = row.get("ROI pixels")
            if vertices is None:
                continue
            verts = np.asarray(vertices, dtype=float)
            if verts.ndim != 2 or len(verts) < 3:
                continue

            color = SEGMENT_COLORS[display_idx % len(SEGMENT_COLORS)]
            patch = Polygon(
                verts,
                closed=True,
                fill=False,
                edgecolor=color,
                linewidth=1.8,
                linestyle="-",
                zorder=3,
                alpha=0.95,
            )
            self.ax_image.add_patch(patch)
            self._saved_roi_overlays.append((patch, row_index))

        self.fig.canvas.draw_idle()

    def _on_area_slider_changed(self) -> None:
        if self._block_area_slider_callbacks:
            return
        self.area_map_cache = None
        self._update_plots()
        if self._active_saved_roi_row_index is not None:
            self._apply_quant_settings_to_all_rows(self._current_quant_settings())
            self._sync_active_quant_row()
        if self.heatmap_enabled:
            self._update_heatmap_display(integrate_only=True)

    def _sync_draw_button(self) -> None:
        if self.roi_tool is None:
            return
        label = "Draw ROI [ON]" if self.roi_tool.draw_mode else "Draw ROI"
        self.btn_draw.label.set_text(label)
        self.fig.canvas.draw_idle()

    def _sync_bg_draw_button(self) -> None:
        if self.bg_roi_tool is None:
            return
        label = "Draw BG [ON]" if self.bg_roi_tool.draw_mode else "Draw BG"
        self.btn_draw_bg.label.set_text(label)
        self.fig.canvas.draw_idle()

    def _on_roi_mode_changed(self) -> None:
        self._sync_draw_button()

    def _on_bg_mode_changed(self) -> None:
        self._sync_bg_draw_button()

    def _clear_roi(self) -> None:
        self._active_saved_roi_row_index = None
        if self.roi_tool is not None:
            self.roi_tool.clear()
        if self.show_saved_rois:
            self._update_saved_roi_display()
        self.fig.canvas.draw_idle()

    def _clear_bg_roi(self) -> None:
        if self.bg_roi_tool is not None:
            self.bg_roi_tool.clear()
        self.raw_bg_trace = None
        self._update_roi_traces()
        self.fig.canvas.draw_idle()

    def _on_save_roi(self, _event) -> None:
        if self.stack is None or self.quant_pickle_path is None:
            self._set_status_message("Cannot save ROI: no stack loaded.")
            return
        if self.roi_tool is None or self.roi_tool.vertices is None or len(self.roi_tool.vertices) < 3:
            self._set_status_message("Cannot save ROI: draw a signal ROI first.")
            return
        if self.raw_trace is None:
            self._set_status_message("Cannot save ROI: ROI trace is not available.")
            return
        if self._active_saved_roi_row_index is not None:
            self._set_status_message(
                "Selected ROI is saved automatically when edited. "
                "Draw a new ROI to add another row."
            )
            return

        store = load_quant_store(self.quant_pickle_path)
        store["rows"].append(
            self._build_quant_row(
                recompute_bleach=True,
                force_auto_fit=False,
                store=store,
            )
        )
        save_quant_store(self.quant_pickle_path, store)
        message = (
            f"ROI saved to {self.quant_pickle_path.name} "
            f"({len(store['rows'])} rows total)."
        )
        self._set_status_message(message)
        if self.show_saved_rois:
            self._update_saved_roi_display()

    def _on_delete_roi(self, _event) -> None:
        if self.stack is None or self.quant_pickle_path is None:
            self._set_status_message("Cannot delete ROI: no stack loaded.")
            return
        if self._active_saved_roi_row_index is None:
            self._set_status_message("Cannot delete ROI: select a saved ROI first.")
            return

        row_index = self._active_saved_roi_row_index
        store = load_quant_store(self.quant_pickle_path)
        if row_index < 0 or row_index >= len(store["rows"]):
            self._active_saved_roi_row_index = None
            self._set_status_message("Cannot delete ROI: row not found.")
            return

        if not messagebox.askyesno(
            "Delete ROI",
            f"Delete saved ROI row {row_index + 1} from {self.quant_pickle_path.name}?",
        ):
            return

        del store["rows"][row_index]
        save_quant_store(self.quant_pickle_path, store)

        self._deselect_saved_roi(clear_traces=False)
        self._load_stack_bg_into_ui(store)
        self._update_roi_traces()
        if self.show_saved_rois:
            self._update_saved_roi_display()
        self._set_status_message(
            f"Deleted ROI row {row_index + 1} from {self.quant_pickle_path.name} "
            f"({len(store['rows'])} rows remaining)."
        )

    def _on_mark_events(self, _event) -> None:
        if self._mark_events_window is not None:
            try:
                if self._mark_events_window.root.winfo_exists():
                    self._mark_events_window.root.lift()
                    self._mark_events_window.root.focus_force()
                    return
            except (tk.TclError, AttributeError):
                pass
            self._mark_events_window = None
        MarkEventsWindow(self)

    def _ensure_area_map_for_mark_events(self) -> np.ndarray | None:
        if self.stack is None or self.z_average is None:
            return None
        if self.pixel_mean_trace is None or self._heatmap_traces_dirty:
            if not self._ensure_pixel_mean_traces():
                return None
        if self.area_map_cache is None and not self._compute_area_map_cache():
            return None
        return self.area_map_cache

    def _set_status_message(self, message: str) -> None:
        self.file_text.set_text(message)
        self.fig.canvas.draw_idle()

    def _build_quant_row(
        self,
        *,
        recompute_bleach: bool = False,
        force_auto_fit: bool = False,
        existing_row: dict | None = None,
        store: dict | None = None,
    ) -> dict:
        width = self.stack.shape[2]
        height = self.stack.shape[1]
        f_left = int(self.slider_area_left.val)
        f_right = int(self.slider_area_right.val)
        if f_right < f_left:
            f_left, f_right = f_right, f_left

        if store is None:
            if self.quant_pickle_path is not None:
                store = load_quant_store(self.quant_pickle_path)
            else:
                store = empty_quant_store()

        bg_pixels, bg_trc = self._resolve_bg_fields_for_quant_row(store)

        roi_vertices = np.asarray(self.roi_tool.vertices, dtype=np.float64)
        max_vals: list[float] = []
        if self.normalized_segments and self.rel_x is not None:
            max_vals = compute_max_vals_per_segment(
                self.normalized_segments, self.rel_x, f_left, f_right
            )

        man_adj = 0.0
        marked_events = None
        bc_auto_shift = format_bc_auto_shift(True)
        if existing_row is not None:
            man_adj = parse_man_adj(existing_row.get("Man. Adj."))
            marked_events = existing_row.get(MARKED_EVENTS_COLUMN)
            bc_auto_shift = existing_row.get(BC_AUTO_SHIFT_COLUMN, format_bc_auto_shift(True))

        row = {
            "directory": os.path.dirname(os.path.abspath(self.file_path)),
            "size": format_stack_size(width, height, self.n_frames),
            "starts": format_start_frames(self.start_frames),
            "freq + avr": format_freq_avr_field(self.acq_fps, self.avr_factor),
            "SG window and order": format_sg_field(
                int(self.slider_window.val), int(self.slider_poly.val)
            ),
            "extension": str(int(self.slider_extension.val)),
            "Area L+R": format_area_lr_field(f_left, f_right),
            "BG pixels": bg_pixels,
            "BG trc": bg_trc,
            "ROI pixels": roi_vertices,
            "ROI trc": np.asarray(self.raw_trace),
            "Area": self._compute_area(f_left, f_right),
            "max vals": np.asarray(max_vals, dtype=np.float64) if max_vals else np.array([]),
            "bleach correct": None,
            "BC baseline": None,
            "fit params": None,
            "BC auto shift": bc_auto_shift,
            "Man. Adj.": man_adj,
            MARKED_EVENTS_COLUMN: marked_events,
            BC_CORR_NORM_TRC_COLUMN: None,
        }
        if recompute_bleach:
            update_row_bleach_correction(row, force_auto_fit=force_auto_fit)
        elif existing_row is not None:
            row[FIT_PARAMS_COLUMN] = existing_row.get(FIT_PARAMS_COLUMN)
            if row.get("BG trc") is not None and row.get("ROI trc") is not None:
                update_row_bleach_correction(row, force_auto_fit=False)
            else:
                row["bleach correct"] = None
                row["BC baseline"] = None
                row[FIT_PARAMS_COLUMN] = None
                row[BC_CORR_NORM_TRC_COLUMN] = None
                return row
            update_row_bc_corr_norm_trc(row)
        return row

    def _on_starts_changed(self, text: str) -> None:
        self.starts_text = text
        self.start_frames = self._parse_starts_from_text(text)
        self._sync_starts_textbox_from_frames(self.start_frames)
        self._update_analysis()

    def _parse_starts_from_text(self, text: str | None = None) -> list[int]:
        raw = self.starts_text if text is None else text
        if self.convert_time_axis:
            return parse_start_seconds(raw, self.n_frames, self._effective_fps())
        return parse_start_frames(raw, self.n_frames)

    def _sync_starts_textbox_from_frames(self, frames: list[int]) -> None:
        if self.convert_time_axis:
            self.starts_text = format_start_seconds(frames, self._effective_fps())
            self.text_starts.label.set_text("Starts (s)")
        else:
            self.starts_text = format_start_frames(frames)
            self.text_starts.label.set_text("Starts")
        self.text_starts.set_val(self.starts_text)

    @staticmethod
    def _parse_positive_float(text: str, default: float) -> float:
        try:
            value = float(text.strip().rstrip("xX").strip())
            return value if value > 0 else default
        except ValueError:
            return default

    def _effective_fps(self) -> float:
        if self.avr_factor <= 0:
            return 0.0
        return self.acq_fps / self.avr_factor

    def _update_effective_fps_display(self) -> None:
        fps = self._effective_fps()
        self.effective_fps_text.set_text(f"{fps:.3f} fps")

    def _on_timing_changed(self, _text: str) -> None:
        self.acq_fps = self._parse_positive_float(self.text_freq.text, DEFAULT_ACQ_FPS)
        self.avr_factor = self._parse_positive_float(self.text_avr.text, float(DEFAULT_AVR))
        self._update_effective_fps_display()
        if self.convert_time_axis:
            self._sync_starts_textbox_from_frames(self.start_frames)
        self._update_plots()

    def _on_convert_toggled(self, _label: str) -> None:
        converting_to_seconds = bool(self.check_convert.get_status()[0])
        if converting_to_seconds:
            frames = parse_start_frames(self.text_starts.text, self.n_frames)
        else:
            frames = parse_start_seconds(
                self.text_starts.text, self.n_frames, self._effective_fps()
            )
        self.start_frames = frames
        self.convert_time_axis = converting_to_seconds
        self._sync_starts_textbox_from_frames(self.start_frames)
        self._update_analysis()

    def _frame_to_axis(self, frame: float, *, one_based: bool = False) -> float:
        if not self.convert_time_axis:
            return frame
        fps = self._effective_fps()
        if fps <= 0:
            return frame
        origin = 1.0 if one_based else 0.0
        return (frame - origin) / fps

    def _frames_to_axis(self, frames: np.ndarray, *, one_based: bool = False) -> np.ndarray:
        values = np.asarray(frames, dtype=float)
        if not self.convert_time_axis:
            return values
        fps = self._effective_fps()
        if fps <= 0:
            return values
        origin = 1.0 if one_based else 0.0
        return (values - origin) / fps

    def _trace_xlabel(self, *, relative: bool = False) -> str:
        if self.convert_time_axis:
            return "Time (s)"
        return "Relative frame" if relative else "Frame"

    def _update_area_slider_limits(self) -> None:
        baseline_len, total_len, _ = segment_geometry(self.extension, self.baseline_fraction)
        self.segment_baseline_len = baseline_len
        self.slider_area_left.valmax = total_len
        self.slider_area_right.valmax = total_len
        self.slider_area_left.ax.set_xlim(1, max(2, total_len))
        self.slider_area_right.ax.set_xlim(1, max(2, total_len))

        signal_start = baseline_len + 1
        self._block_area_slider_callbacks = True
        try:
            if self.slider_area_left.val < 1 or self.slider_area_left.val > total_len:
                self.slider_area_left.set_val(signal_start)
            if self.slider_area_right.val > total_len or self.slider_area_right.val <= self.slider_area_left.val:
                self.slider_area_right.set_val(total_len)
        finally:
            self._block_area_slider_callbacks = False

    def load_stack(self, path: str) -> None:
        try:
            stack = load_tif_stack(path)
        except (OSError, ValueError) as exc:
            self.file_text.set_text(f"Failed to load: {exc}")
            self.fig.canvas.draw_idle()
            return

        self.stack = stack
        self.file_path = path
        self.quant_pickle_path = ensure_quant_pickle(path)
        self.n_frames = stack.shape[0]
        self.z_average = compute_z_average(stack)
        self._heatmap_traces_dirty = True
        self.pixel_mean_trace = None
        self.pixel_rel_x = None
        self.area_map_cache = None

        height, width = self.z_average.shape
        self._active_saved_roi_row_index = None
        self._refresh_base_image()

        self.roi_tool = EditableROI(
            self.ax_image,
            (height, width),
            self._on_roi_changed,
            on_mode_change=self._on_roi_mode_changed,
            edge_color="lime",
            zorder=5,
            on_press_intercept=self._saved_roi_press_intercept,
        )
        self.bg_roi_tool = EditableROI(
            self.ax_image,
            (height, width),
            self._on_bg_changed,
            on_mode_change=self._on_bg_mode_changed,
            edge_color="#1e88e5",
            fill_color="#1e88e5",
            fill_alpha=0.5,
            handle_face_color="#bbdefb",
            preview_color="#1e88e5",
            zorder=4,
            sibling=self.roi_tool,
            enable_delete_key=False,
        )
        self.roi_tool.sibling = self.bg_roi_tool
        self.raw_bg_trace = None

        self.slider_window.valmax = max(3, self.n_frames if self.n_frames % 2 else self.n_frames - 1)
        if self.slider_window.val > self.slider_window.valmax:
            self.slider_window.set_val(min(51, self.slider_window.valmax))

        store = load_quant_store(self.quant_pickle_path)
        store_changed = backfill_bleach_correction(store) or backfill_bc_corr_norm_trc(store)
        if self._ensure_unified_bg_on_load(store):
            store_changed = True
        if store_changed:
            save_quant_store(self.quant_pickle_path, store)
        self._load_stack_bg_into_ui(store)
        pickle_settings = self._pickle_settings_for_stack(store)
        if pickle_settings is not None:
            self._apply_quant_settings_to_gui(pickle_settings)
        else:
            self.start_frames = parse_start_frames(DEFAULT_STARTS, self.n_frames)
            self._sync_starts_textbox_from_frames(self.start_frames)
            self._update_area_slider_limits()
            self._update_analysis()

        name = path if len(path) <= 120 else "…" + path[-117:]
        self.file_text.set_text(f"{name}  |  {self.n_frames} frames, {height}×{width}")

        if self.show_saved_rois:
            self._update_saved_roi_display()

    def _mark_heatmap_dirty(self) -> None:
        self._heatmap_traces_dirty = True
        self.pixel_mean_trace = None
        self.pixel_rel_x = None
        self.area_map_cache = None

    def _refresh_base_image(self) -> None:
        if self.z_average is None:
            return
        height, width = self.z_average.shape
        self._clear_heatmap_layers()
        if self.base_image is not None:
            try:
                self.base_image.remove()
            except (ValueError, AttributeError):
                pass
            self.base_image = None
        self.base_image = self.ax_image.imshow(
            self.z_average,
            cmap="gray",
            extent=[0, width, height, 0],
            aspect="equal",
            zorder=1,
        )
        self.ax_image.set_title("Z-average")
        self.ax_image.set_xlim(0, width)
        self.ax_image.set_ylim(height, 0)
        self._redraw_rois()
        if self.show_saved_rois:
            self._update_saved_roi_display()

    def _redraw_rois(self) -> None:
        for tool in (self.roi_tool, self.bg_roi_tool):
            if tool is not None:
                tool._update_patch()

    def _clear_heatmap_progress(self) -> None:
        for artist in self._heatmap_progress_artists:
            try:
                artist.remove()
            except (ValueError, AttributeError):
                pass
        self._heatmap_progress_artists = []

    def _safe_remove_heatmap_overlay(self) -> None:
        if self.heatmap_overlay is None:
            return
        try:
            self.heatmap_overlay.remove()
        except (ValueError, AttributeError):
            pass
        self.heatmap_overlay = None

    def _hide_heatmap_colorbar_axis(self) -> None:
        self.ax_heatmap_cbar.cla()
        self.ax_heatmap_cbar.set_axis_off()

    def _prepare_heatmap_colorbar_axis(self) -> None:
        self.ax_heatmap_cbar.cla()
        self.ax_heatmap_cbar.set_axis_on()

    def _safe_remove_heatmap_colorbar(self) -> None:
        colorbar = self.heatmap_colorbar
        self.heatmap_colorbar = None
        if colorbar is not None:
            try:
                colorbar.remove()
            except (KeyError, ValueError, AttributeError):
                pass
        self._hide_heatmap_colorbar_axis()

    def _clear_heatmap_layers(self) -> None:
        self._safe_remove_heatmap_overlay()
        self._safe_remove_heatmap_colorbar()
        self._clear_heatmap_progress()

    def _ensure_pixel_mean_traces(self) -> bool:
        if self.stack is None:
            return False
        if not self._heatmap_traces_dirty and self.pixel_mean_trace is not None:
            return True

        starts = self.start_frames
        window = int(self.slider_window.val)
        poly = int(self.slider_poly.val)
        extension = int(self.slider_extension.val)

        def report_progress(stage: str, fraction: float) -> None:
            self._set_heatmap_progress(stage, fraction)

        self._set_heatmap_progress("Starting", 0.0)

        self._block_area_slider_callbacks = True
        try:
            result = compute_all_pixel_mean_traces(
                self.stack,
                starts,
                extension,
                window,
                poly,
                self.baseline_fraction,
                progress=report_progress,
            )
        finally:
            self._block_area_slider_callbacks = False
        if result is None:
            self.pixel_mean_trace = None
            self.pixel_rel_x = None
            self._heatmap_traces_dirty = False
            return False

        self.pixel_rel_x, self.pixel_mean_trace = result
        self._heatmap_traces_dirty = False
        return True

    def _set_heatmap_progress(self, stage: str, fraction: float) -> None:
        fraction = max(0.0, min(1.0, fraction))
        pct = int(round(fraction * 100))
        self._clear_heatmap_progress()

        bar_left, bar_width, bar_height, bar_bottom = 0.08, 0.84, 0.08, 0.46
        self._heatmap_progress_artists.append(
            self.ax_image.add_patch(
                Rectangle(
                    (bar_left, bar_bottom),
                    bar_width,
                    bar_height,
                    transform=self.ax_image.transAxes,
                    facecolor="0.88",
                    edgecolor="0.65",
                    linewidth=1,
                    zorder=20,
                )
            )
        )
        if fraction > 0:
            self._heatmap_progress_artists.append(
                self.ax_image.add_patch(
                    Rectangle(
                        (bar_left, bar_bottom),
                        bar_width * fraction,
                        bar_height,
                        transform=self.ax_image.transAxes,
                        facecolor="#27ae60",
                        edgecolor="none",
                        zorder=21,
                    )
                )
            )
        self._heatmap_progress_artists.append(
            self.ax_image.text(
                0.5,
                0.30,
                stage,
                transform=self.ax_image.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                zorder=22,
            )
        )
        self._heatmap_progress_artists.append(
            self.ax_image.text(
                0.5,
                0.18,
                f"{pct}% complete",
                transform=self.ax_image.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="0.35",
                zorder=22,
            )
        )
        self.ax_image.set_title(f"Z-average — heatmap {stage} ({pct}%)")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _finalize_heatmap_render(self) -> None:
        self._clear_heatmap_progress()
        self.ax_image.set_title("Z-average + area heatmap")
        self.fig.canvas.draw_idle()

    def _compute_area_map_cache(self) -> bool:
        if self.pixel_mean_trace is None or self.pixel_rel_x is None:
            return False

        height, width = self.stack.shape[1], self.stack.shape[2]
        f_left = int(self.slider_area_left.val)
        f_right = int(self.slider_area_right.val)
        area_map = compute_area_from_mean_trace(
            self.pixel_rel_x,
            self.pixel_mean_trace.reshape(self.pixel_mean_trace.shape[0], -1),
            f_left,
            f_right,
        )
        self.area_map_cache = np.asarray(area_map, dtype=np.float64).reshape(height, width)
        return True

    def _update_heatmap_overlay_inplace(self) -> bool:
        if self.area_map_cache is None or self.heatmap_overlay is None:
            return False

        self._clear_heatmap_progress()
        data = self.area_map_cache
        self.heatmap_overlay.set_data(data)
        vmin = float(np.nanmin(data))
        vmax = float(np.nanmax(data))
        if vmin >= vmax:
            vmax = vmin + 1.0
        self.heatmap_overlay.set_clim(vmin, vmax)
        if self.heatmap_colorbar is not None:
            self.heatmap_colorbar.update_normal(self.heatmap_overlay)
        self.ax_image.set_title("Z-average + area heatmap")
        if self.roi_tool is not None:
            self.roi_tool._update_patch()
        if self.bg_roi_tool is not None:
            self.bg_roi_tool._update_patch()
        self.fig.canvas.draw_idle()
        return True

    def _show_heatmap_overlay(self) -> None:
        if self.area_map_cache is None or self.z_average is None:
            return

        if self.heatmap_overlay is not None and self._update_heatmap_overlay_inplace():
            self._finalize_heatmap_render()
            return

        height, width = self.z_average.shape
        self._safe_remove_heatmap_overlay()
        self._safe_remove_heatmap_colorbar()

        if self.base_image is None:
            self._refresh_base_image()

        self.heatmap_overlay = self.ax_image.imshow(
            self.area_map_cache,
            cmap="inferno",
            alpha=0.5,
            extent=[0, width, height, 0],
            aspect="equal",
            zorder=2,
        )
        self._prepare_heatmap_colorbar_axis()
        self.heatmap_colorbar = self.fig.colorbar(
            self.heatmap_overlay, cax=self.ax_heatmap_cbar
        )
        if self.roi_tool is not None:
            self.roi_tool._update_patch()
        if self.bg_roi_tool is not None:
            self.bg_roi_tool._update_patch()
        self._finalize_heatmap_render()

    def _update_heatmap_display(self, integrate_only: bool = False) -> None:
        if self._heatmap_busy:
            self._heatmap_pending = True
            if not integrate_only:
                self._heatmap_pending_full = True
            return
        self._heatmap_busy = True
        try:
            self._update_heatmap_display_impl(integrate_only)
        finally:
            self._heatmap_busy = False
            if self._heatmap_pending:
                pending_full = self._heatmap_pending_full
                self._heatmap_pending = False
                self._heatmap_pending_full = False
                self._update_heatmap_display(integrate_only=not pending_full)

    def _update_heatmap_display_impl(self, integrate_only: bool = False) -> None:
        if not self.heatmap_enabled or self.stack is None:
            self._clear_heatmap_layers()
            self.ax_image.set_title("Z-average")
            self.fig.canvas.draw_idle()
            return

        if self.base_image is None:
            self._refresh_base_image()

        need_traces = (
            not integrate_only
            and (self._heatmap_traces_dirty or self.pixel_mean_trace is None)
        )
        if integrate_only and self.pixel_mean_trace is None:
            need_traces = True

        if need_traces:
            if not self._ensure_pixel_mean_traces():
                self._clear_heatmap_layers()
                self.ax_image.set_title("Z-average (heatmap: no valid segments)")
                self.fig.canvas.draw_idle()
                return

        if self.pixel_mean_trace is None or self.pixel_rel_x is None:
            self._clear_heatmap_layers()
            self.ax_image.set_title("Z-average (heatmap: no valid segments)")
            self.fig.canvas.draw_idle()
            return

        if integrate_only:
            if self.heatmap_overlay is not None and not self._heatmap_traces_dirty:
                if not self._compute_area_map_cache():
                    self._clear_heatmap_layers()
                    self.ax_image.set_title("Z-average (heatmap: no valid segments)")
                    self.fig.canvas.draw_idle()
                    return
                self._update_heatmap_overlay_inplace()
                return

            if not self._compute_area_map_cache():
                self._clear_heatmap_layers()
                self.ax_image.set_title("Z-average (heatmap: no valid segments)")
                self.fig.canvas.draw_idle()
                return
            self._show_heatmap_overlay()
            return

        if self.area_map_cache is None:
            self._set_heatmap_progress("Integrating area map", 0.96)
            if not self._compute_area_map_cache():
                self._clear_heatmap_layers()
                self.ax_image.set_title("Z-average (heatmap: no valid segments)")
                self.fig.canvas.draw_idle()
                return

        self._set_heatmap_progress("Rendering heatmap", 0.99)
        self._show_heatmap_overlay()

    def _on_roi_changed(self) -> None:
        if self.stack is None or self.roi_tool is None:
            return
        self.raw_trace = compute_raw_trace(self.stack, self.roi_tool.mask)
        if not self._loading_saved_roi and self.roi_tool.vertices is None:
            self._active_saved_roi_row_index = None
        self._update_roi_traces()
        if self._active_saved_roi_row_index is not None and not self._loading_saved_roi:
            self._apply_quant_settings_to_all_rows(self._current_quant_settings())
            self._sync_active_quant_row()
        if self.show_saved_rois:
            self._update_saved_roi_display()

    def _on_bg_changed(self) -> None:
        if self.stack is None or self.bg_roi_tool is None:
            return

        if self._suppress_bg_change_prompt:
            mask = self.bg_roi_tool.mask
            if mask is None or not np.any(mask):
                self.raw_bg_trace = None
            else:
                self.raw_bg_trace = compute_raw_trace(self.stack, mask)
            self._update_roi_traces()
            return

        store = load_quant_store(self.quant_pickle_path) if self.quant_pickle_path else None
        bg_row = self._first_stack_row_with_bg(store) if store is not None else None
        canonical_pixels = None if bg_row is None else bg_row.get("BG pixels")

        mask = self.bg_roi_tool.mask
        new_vertices = (
            None
            if self.bg_roi_tool.vertices is None
            else np.asarray(self.bg_roi_tool.vertices, dtype=np.float64)
        )
        has_bg = (
            new_vertices is not None
            and mask is not None
            and np.any(mask)
            and len(new_vertices) >= 3
        )

        if not has_bg:
            if bg_pixels_equal(canonical_pixels, None):
                self.raw_bg_trace = None
            elif self._ask_apply_new_bg_roi():
                self.raw_bg_trace = None
                self._apply_bg_to_all_stack_rows(None, None)
            else:
                self._restore_canonical_bg_roi()
                return
        elif bg_pixels_equal(canonical_pixels, new_vertices):
            self.raw_bg_trace = compute_raw_trace(self.stack, mask)
        elif self._ask_apply_new_bg_roi():
            self.raw_bg_trace = compute_raw_trace(self.stack, mask)
            self._apply_bg_to_all_stack_rows(new_vertices, self.raw_bg_trace)
        else:
            self._restore_canonical_bg_roi()
            return

        if self._active_saved_roi_row_index is not None and not self._loading_saved_roi:
            self._apply_quant_settings_to_all_rows(self._current_quant_settings())
            self._sync_active_quant_row()
        self._update_roi_traces()

    def _corrected_smooth_trace(self) -> np.ndarray | None:
        if self.raw_trace is None:
            return None

        window = int(self.slider_window.val)
        poly = int(self.slider_poly.val)
        smooth_signal = apply_savgol(self.raw_trace, window, poly)
        if self.raw_bg_trace is None:
            return smooth_signal

        smooth_bg = apply_savgol(self.raw_bg_trace, window, poly)
        return smooth_signal - smooth_bg

    def _saved_row_for_bleach_display(self) -> dict | None:
        if self._active_saved_roi_row_index is None or self.quant_pickle_path is None:
            return None
        store = load_quant_store(self.quant_pickle_path)
        row_index = self._active_saved_roi_row_index
        rows = store.get("rows", [])
        if 0 <= row_index < len(rows):
            return rows[row_index]
        return None

    def _compute_bc_baseline_trace(self) -> np.ndarray | None:
        if self.raw_trace is None or self.raw_bg_trace is None:
            return None

        window = int(self.slider_window.val)
        poly = int(self.slider_poly.val)
        smooth = compute_bg_corrected_smooth_trace(
            self.raw_trace, self.raw_bg_trace, window, poly
        )
        return compute_bc_baseline_for_smooth(smooth, self._saved_row_for_bleach_display())

    def _update_roi_traces(self) -> None:
        """Refresh ROI-based traces/plots only; heatmap is full-image and unchanged."""
        if self.stack is None:
            return

        self.extension = int(self.slider_extension.val)

        self.smooth_trace = self._corrected_smooth_trace()
        self.bc_baseline_trace = self._compute_bc_baseline_trace()
        self._compute_mean_trace()
        self._update_plots()

    def _update_analysis(self) -> None:
        self._mark_heatmap_dirty()
        self._update_roi_traces()
        self._redraw_rois()
        if self.heatmap_enabled:
            self._update_heatmap_display()

    def _compute_mean_trace(self) -> None:
        self.rel_x = None
        self.normalized_segments = []
        self.mean_trace_values = None
        self.baseline_level = 1.0

        if self.smooth_trace is None or not self.start_frames:
            self._update_area_slider_limits()
            return

        result = build_normalized_segments(
            self.smooth_trace,
            self.start_frames,
            self.extension,
            self.baseline_fraction,
        )
        self._update_area_slider_limits()

        if result is None:
            return

        rel_x, normalized_segments, mean_normalized, valid_starts, baseline_len = result
        self.rel_x = rel_x
        self.normalized_segments = normalized_segments
        self.mean_trace_values = mean_normalized
        self.start_frames = valid_starts
        self.segment_baseline_len = baseline_len

    def _update_plots(self) -> None:
        n_frames = self.n_frames
        stack_frames = np.arange(n_frames, dtype=float) if n_frames else np.array([])
        x = self._frames_to_axis(stack_frames, one_based=False)

        self.ax_raw.clear()
        if n_frames:
            if self.raw_trace is not None:
                self.ax_raw.plot(x, self.raw_trace, color="0.25", linewidth=0.8, label="ROI")
            if self.raw_bg_trace is not None:
                self.ax_raw.plot(x, self.raw_bg_trace, color="#1e88e5", linewidth=0.8, label="BG")
            if self.raw_trace is not None or self.raw_bg_trace is not None:
                self.ax_raw.legend(loc="upper right", fontsize=8)
        self.ax_raw.set_ylabel("Intensity")
        self.ax_raw.set_title("raw")
        self.ax_raw.set_xlabel(self._trace_xlabel())
        if n_frames:
            self.ax_raw.set_xlim(x[0], x[-1])

        self.ax_smooth.clear()
        if self.smooth_trace is not None and n_frames:
            self.ax_smooth.plot(x, self.smooth_trace, color="0.15", linewidth=1.0, label="smoothed")
            if self.bc_baseline_trace is not None:
                bc_len = len(self.bc_baseline_trace)
                self.ax_smooth.plot(
                    x[:bc_len],
                    self.bc_baseline_trace,
                    color="red",
                    linestyle=":",
                    linewidth=1.2,
                    label="BC baseline",
                )
            extension = self.extension
            for idx, start in enumerate(self.start_frames):
                if start + extension > n_frames:
                    continue
                color = SEGMENT_COLORS[idx % len(SEGMENT_COLORS)]
                span_left = self._frame_to_axis(start, one_based=False)
                span_right = self._frame_to_axis(start + extension, one_based=False)
                self.ax_smooth.axvspan(span_left, span_right, color=color, alpha=0.18, lw=0)
            if self.bc_baseline_trace is not None:
                self.ax_smooth.legend(loc="upper right", fontsize=8)
        self.ax_smooth.set_ylabel("Intensity")
        title = "smoothed (ROI − BG)" if self.raw_bg_trace is not None else "smoothed"
        self.ax_smooth.set_title(title)
        self.ax_smooth.set_xlabel(self._trace_xlabel())

        self.ax_segments.clear()
        baseline_len, total_len, _ = segment_geometry(self.extension, self.baseline_fraction)

        if self.smooth_trace is not None and self.rel_x is not None and self.normalized_segments:
            rel_x = self.rel_x
            seg_x = self._frames_to_axis(rel_x, one_based=True)
            for idx, seg_y in enumerate(self.normalized_segments):
                color = SEGMENT_COLORS[idx % len(SEGMENT_COLORS)]
                self.ax_segments.plot(seg_x, seg_y, color=color, linewidth=1.0, alpha=0.85)

            if self.mean_trace_values is not None:
                self.ax_segments.plot(
                    seg_x,
                    self.mean_trace_values,
                    color="black",
                    linewidth=3.0,
                    label="mean",
                    zorder=5,
                )

            self.ax_segments.axhline(
                self.baseline_level,
                color="red",
                linestyle=":",
                linewidth=1.5,
                label="baseline (=1)",
            )
            self.ax_segments.axvline(
                self._frame_to_axis(baseline_len + 0.5, one_based=True),
                color="0.7",
                linestyle="-",
                linewidth=0.8,
                alpha=0.6,
            )

            f_left = int(self.slider_area_left.val)
            f_right = int(self.slider_area_right.val)
            if f_right < f_left:
                f_left, f_right = f_right, f_left

            self.ax_segments.axvline(
                self._frame_to_axis(f_left, one_based=True),
                color="0.4",
                linestyle="--",
                linewidth=1.2,
            )
            self.ax_segments.axvline(
                self._frame_to_axis(f_right, one_based=True),
                color="0.4",
                linestyle="--",
                linewidth=1.2,
            )

            self.computed_area = self._compute_area(f_left, f_right)
            if self.mean_trace_values is not None:
                mask = (rel_x >= f_left) & (rel_x <= f_right)
                if np.any(mask):
                    area_x = seg_x[mask]
                    area_y = self.mean_trace_values[mask]
                    self.ax_segments.fill_between(
                        area_x,
                        self.baseline_level,
                        area_y,
                        where=area_y >= self.baseline_level,
                        color="0.75",
                        alpha=0.35,
                    )

            self.ax_segments.text(
                0.02,
                0.95,
                f"Area = {self.computed_area:.3f}",
                transform=self.ax_segments.transAxes,
                va="top",
                fontsize=10,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )
            self.ax_segments.set_xlim(seg_x[0], seg_x[-1])

        self.ax_segments.set_ylabel("Normalized")
        self.ax_segments.set_xlabel(self._trace_xlabel(relative=True))
        self.ax_segments.set_title("segments (overlaid)")

        self.fig.canvas.draw_idle()

    def _compute_area(self, f_left: int, f_right: int) -> float:
        if self.rel_x is None or self.mean_trace_values is None:
            return 0.0
        return float(
            compute_area_from_mean_trace(
                self.rel_x,
                self.mean_trace_values,
                f_left,
                f_right,
                self.baseline_level,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive TIFF stack ROI analyzer")
    parser.add_argument("stack", nargs="?", help="Optional path to a .tif stack")
    args = parser.parse_args()

    app = StackAnalyzerApp(initial_path=args.stack)
    plt.show()


if __name__ == "__main__":
    main()
