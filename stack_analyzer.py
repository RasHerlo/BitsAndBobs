#!/usr/bin/env python3
"""Interactive GUI for TIFF stack ROI fluorescence trace analysis."""

from __future__ import annotations

import argparse
import tkinter as tk
from collections.abc import Callable
from tkinter import filedialog

import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np
import tifffile
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon, Rectangle
from scipy.integrate import trapezoid
from scipy.signal import savgol_filter

SEGMENT_COLORS = plt.cm.tab10.colors
HANDLE_RADIUS = 14
MIN_FREEHAND_POINTS = 4
MIN_POINT_SPACING = 1.5
MAX_EDIT_VERTICES = 64
DEFAULT_STARTS = "896, 1050, 1205, 1359, 1513"


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


class EditableROI:
    """Freehand polygon ROI: draw, move, vertex-adjust, delete."""

    def __init__(self, ax, image_shape: tuple[int, int], on_change, on_mode_change=None):
        self.ax = ax
        self.height, self.width = image_shape
        self.on_change = on_change
        self.on_mode_change = on_mode_change

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
        self._remove_artist(self.patch)
        self.patch = None
        self._remove_artist(self.handle_artists)
        self.handle_artists = None
        self._remove_artist(self.preview_line)
        self.preview_line = None
        self.on_change()

    def _remove_artist(self, artist) -> None:
        if artist is not None:
            artist.remove()

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
            self._remove_artist(self.patch)
            self.patch = None
            self._update_handles()
            return

        if self.patch is None:
            self.patch = Polygon(
                self.vertices,
                closed=True,
                fill=False,
                edgecolor="lime",
                linewidth=2,
                zorder=5,
            )
            self.ax.add_patch(self.patch)
        else:
            self.patch.set_xy(self.vertices)
        self._update_handles()

    def _update_handles(self) -> None:
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
            markerfacecolor="yellow",
            markeredgecolor="lime",
            markeredgewidth=1,
            zorder=6,
        )

    def _update_preview(self) -> None:
        if not self._current_stroke:
            return

        xs, ys = zip(*self._current_stroke)
        if self.preview_line is None:
            (self.preview_line,) = self.ax.plot(xs, ys, color="lime", linewidth=2, alpha=0.9)
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

    def _on_press(self, event) -> None:
        if event.button != 1:
            return

        xy = self._event_xy(event) if event.inaxes is self.ax else None
        if xy is None:
            return
        x, y = xy

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
        if event.key in ("delete", "backspace"):
            self.clear()
            self.ax.figure.canvas.draw_idle()


class StackAnalyzerApp:
    def __init__(self, initial_path: str | None = None):
        self.stack: np.ndarray | None = None
        self.z_average: np.ndarray | None = None
        self.raw_trace: np.ndarray | None = None
        self.smooth_trace: np.ndarray | None = None
        self.file_path = initial_path or ""

        self.n_frames = 0
        self.extension = 50
        self.baseline_fraction = 0.2
        self.start_frames = [0]
        self.starts_text = DEFAULT_STARTS
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

        left_gs = gs[1:4, 0].subgridspec(2, 1, height_ratios=[0.90, 0.10], hspace=0.08)
        self.ax_image = self.fig.add_subplot(left_gs[0, 0])
        toggle_ax = self.fig.add_subplot(left_gs[1, 0])
        toggle_ax.set_axis_off()
        self.check_heatmap = widgets.CheckButtons(toggle_ax, ["Heatmap"], [False])
        self.check_heatmap.on_clicked(self._on_heatmap_toggled)

        self.ax_raw = self.fig.add_subplot(gs[1, 1:3])
        self.ax_smooth = self.fig.add_subplot(gs[2, 1:3], sharex=self.ax_raw)
        self.ax_segments = self.fig.add_subplot(gs[3, 1:3])

        self._build_controls()
        self.roi_tool: EditableROI | None = None

        if initial_path:
            self.load_stack(initial_path)

    def _build_controls(self) -> None:
        self.file_text = self.fig.text(
            0.05, 0.975, "No file loaded", fontsize=9, va="top", ha="left"
        )

        ax_browse = self.fig.add_axes([0.05, 0.905, 0.08, 0.035])
        self.btn_browse = widgets.Button(ax_browse, "Browse…")
        self.btn_browse.on_clicked(self._browse_file)

        ax_draw = self.fig.add_axes([0.05, 0.855, 0.08, 0.035])
        self.btn_draw = widgets.Button(ax_draw, "Draw ROI")
        self.btn_draw.on_clicked(self._toggle_draw_mode)

        ax_clear = self.fig.add_axes([0.14, 0.855, 0.08, 0.035])
        self.btn_clear = widgets.Button(ax_clear, "Clear ROI")
        self.btn_clear.on_clicked(lambda _event: self._clear_roi())

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

        ax_area_left = self.fig.add_axes([0.74, 0.905, 0.18, 0.025])
        self.slider_area_left = widgets.Slider(ax_area_left, "Area L", 1, 60, valinit=1, valstep=1)
        self.slider_area_left.on_changed(lambda _val: self._on_area_slider_changed())

        ax_area_right = self.fig.add_axes([0.74, 0.855, 0.18, 0.025])
        self.slider_area_right = widgets.Slider(ax_area_right, "Area R", 1, 60, valinit=60, valstep=1)
        self.slider_area_right.on_changed(lambda _val: self._on_area_slider_changed())

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

    def _toggle_draw_mode(self, _event) -> None:
        if self.roi_tool is None:
            return
        self.roi_tool.set_draw_mode(not self.roi_tool.draw_mode)
        self._sync_draw_button()

    def _on_heatmap_toggled(self, _label: str) -> None:
        self.heatmap_enabled = bool(self.check_heatmap.get_status()[0])
        self._update_heatmap_display()

    def _on_area_slider_changed(self) -> None:
        self.area_map_cache = None
        self._update_plots()
        if self.heatmap_enabled:
            self._update_heatmap_display(integrate_only=True)

    def _sync_draw_button(self) -> None:
        if self.roi_tool is None:
            return
        label = "Draw ROI [ON]" if self.roi_tool.draw_mode else "Draw ROI"
        self.btn_draw.label.set_text(label)
        self.fig.canvas.draw_idle()

    def _on_roi_mode_changed(self) -> None:
        self._sync_draw_button()

    def _clear_roi(self) -> None:
        if self.roi_tool is not None:
            self.roi_tool.clear()
            self.fig.canvas.draw_idle()

    def _on_starts_changed(self, text: str) -> None:
        self.starts_text = text
        self._update_analysis()

    def _update_area_slider_limits(self) -> None:
        baseline_len, total_len, _ = segment_geometry(self.extension, self.baseline_fraction)
        self.segment_baseline_len = baseline_len
        self.slider_area_left.valmax = total_len
        self.slider_area_right.valmax = total_len
        self.slider_area_left.ax.set_xlim(1, max(2, total_len))
        self.slider_area_right.ax.set_xlim(1, max(2, total_len))

        signal_start = baseline_len + 1
        if self.slider_area_left.val < 1 or self.slider_area_left.val > total_len:
            self.slider_area_left.set_val(signal_start)
        if self.slider_area_right.val > total_len or self.slider_area_right.val <= self.slider_area_left.val:
            self.slider_area_right.set_val(total_len)

    def load_stack(self, path: str) -> None:
        try:
            stack = load_tif_stack(path)
        except (OSError, ValueError) as exc:
            self.file_text.set_text(f"Failed to load: {exc}")
            self.fig.canvas.draw_idle()
            return

        self.stack = stack
        self.file_path = path
        self.n_frames = stack.shape[0]
        self.z_average = compute_z_average(stack)
        self._heatmap_traces_dirty = True
        self.pixel_mean_trace = None
        self.pixel_rel_x = None
        self.area_map_cache = None

        height, width = self.z_average.shape
        self._refresh_base_image()

        self.roi_tool = EditableROI(
            self.ax_image,
            (height, width),
            self._on_roi_changed,
            on_mode_change=self._on_roi_mode_changed,
        )

        self.slider_window.valmax = max(3, self.n_frames if self.n_frames % 2 else self.n_frames - 1)
        if self.slider_window.val > self.slider_window.valmax:
            self.slider_window.set_val(min(51, self.slider_window.valmax))

        self.starts_text = DEFAULT_STARTS
        self.text_starts.set_val(DEFAULT_STARTS)
        self._update_area_slider_limits()

        name = path if len(path) <= 120 else "…" + path[-117:]
        self.file_text.set_text(f"{name}  |  {self.n_frames} frames, {height}×{width}")

        self._update_analysis()

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
        self.ax_image.clear()
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
        if self.roi_tool is not None:
            self.roi_tool._update_patch()

    def _clear_heatmap_progress(self) -> None:
        for artist in self._heatmap_progress_artists:
            artist.remove()
        self._heatmap_progress_artists = []

    def _clear_heatmap_layers(self) -> None:
        if self.heatmap_overlay is not None:
            self.heatmap_overlay.remove()
            self.heatmap_overlay = None
        if self.heatmap_colorbar is not None:
            self.heatmap_colorbar.remove()
            self.heatmap_colorbar = None
        self._clear_heatmap_progress()

    def _ensure_pixel_mean_traces(self) -> bool:
        if self.stack is None:
            return False
        if not self._heatmap_traces_dirty and self.pixel_mean_trace is not None:
            return True

        starts = parse_start_frames(self.starts_text, self.n_frames)
        window = int(self.slider_window.val)
        poly = int(self.slider_poly.val)
        extension = int(self.slider_extension.val)

        def report_progress(stage: str, fraction: float) -> None:
            self._set_heatmap_progress(stage, fraction)
            self.fig.canvas.flush_events()

        self._set_heatmap_progress("Starting", 0.0)
        self.fig.canvas.flush_events()

        result = compute_all_pixel_mean_traces(
            self.stack,
            starts,
            extension,
            window,
            poly,
            self.baseline_fraction,
            progress=report_progress,
        )
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

    def _show_heatmap_overlay(self) -> None:
        if self.area_map_cache is None or self.z_average is None:
            return

        height, width = self.z_average.shape
        self._clear_heatmap_layers()

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
        self.ax_image.set_title("Z-average + area heatmap")
        self.heatmap_colorbar = self.fig.colorbar(
            self.heatmap_overlay, ax=self.ax_image, fraction=0.046, pad=0.04
        )
        if self.roi_tool is not None:
            self.roi_tool._update_patch()
        self.fig.canvas.draw_idle()

    def _update_heatmap_display(self, integrate_only: bool = False) -> None:
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

        if self.area_map_cache is None or integrate_only:
            self._set_heatmap_progress("Integrating area map", 0.96)
            self.fig.canvas.flush_events()
            if not self._compute_area_map_cache():
                self._clear_heatmap_layers()
                self.ax_image.set_title("Z-average (heatmap: no valid segments)")
                self.fig.canvas.draw_idle()
                return

        self._clear_heatmap_progress()
        self._set_heatmap_progress("Rendering heatmap", 0.99)
        self.fig.canvas.flush_events()
        self._clear_heatmap_progress()
        self._show_heatmap_overlay()

    def _on_roi_changed(self) -> None:
        if self.stack is None or self.roi_tool is None:
            return
        self.raw_trace = compute_raw_trace(self.stack, self.roi_tool.mask)
        self._update_roi_traces()

    def _update_roi_traces(self) -> None:
        """Refresh ROI-based traces/plots only; heatmap is full-image and unchanged."""
        if self.stack is None:
            return

        window = int(self.slider_window.val)
        poly = int(self.slider_poly.val)
        self.extension = int(self.slider_extension.val)
        if self.n_frames > 0:
            self.start_frames = parse_start_frames(self.starts_text, self.n_frames)

        if self.raw_trace is not None:
            self.smooth_trace = apply_savgol(self.raw_trace, window, poly)
        else:
            self.smooth_trace = None

        self._compute_mean_trace()
        self._update_plots()

    def _update_analysis(self) -> None:
        self._mark_heatmap_dirty()
        self._update_roi_traces()
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
        x = np.arange(n_frames) if n_frames else np.array([])

        self.ax_raw.clear()
        if self.raw_trace is not None and n_frames:
            self.ax_raw.plot(x, self.raw_trace, color="0.25", linewidth=0.8)
        self.ax_raw.set_ylabel("Intensity")
        self.ax_raw.set_title("raw")
        if n_frames:
            self.ax_raw.set_xlim(0, n_frames - 1)

        self.ax_smooth.clear()
        if self.smooth_trace is not None and n_frames:
            self.ax_smooth.plot(x, self.smooth_trace, color="0.15", linewidth=1.0)
            extension = self.extension
            baseline_len = max(1, int(round(self.baseline_fraction * extension)))
            for idx, start in enumerate(self.start_frames):
                if start + extension > n_frames:
                    continue
                color = SEGMENT_COLORS[idx % len(SEGMENT_COLORS)]
                self.ax_smooth.axvspan(start, start + extension, color=color, alpha=0.18, lw=0)
        self.ax_smooth.set_ylabel("Intensity")
        self.ax_smooth.set_title("smoothed")

        self.ax_segments.clear()
        baseline_len, total_len, _ = segment_geometry(self.extension, self.baseline_fraction)

        if self.smooth_trace is not None and self.rel_x is not None and self.normalized_segments:
            rel_x = self.rel_x
            for idx, seg_y in enumerate(self.normalized_segments):
                color = SEGMENT_COLORS[idx % len(SEGMENT_COLORS)]
                self.ax_segments.plot(rel_x, seg_y, color=color, linewidth=1.0, alpha=0.85)

            if self.mean_trace_values is not None:
                self.ax_segments.plot(
                    rel_x,
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
                baseline_len + 0.5,
                color="0.7",
                linestyle="-",
                linewidth=0.8,
                alpha=0.6,
            )

            f_left = int(self.slider_area_left.val)
            f_right = int(self.slider_area_right.val)
            if f_right < f_left:
                f_left, f_right = f_right, f_left

            self.ax_segments.axvline(f_left, color="0.4", linestyle="--", linewidth=1.2)
            self.ax_segments.axvline(f_right, color="0.4", linestyle="--", linewidth=1.2)

            self.computed_area = self._compute_area(f_left, f_right)
            if self.mean_trace_values is not None:
                mask = (rel_x >= f_left) & (rel_x <= f_right)
                if np.any(mask):
                    area_x = rel_x[mask]
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
            self.ax_segments.set_xlim(1, total_len)

        self.ax_segments.set_ylabel("Normalized")
        self.ax_segments.set_xlabel("Relative frame")
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
