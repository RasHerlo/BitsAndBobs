#!/usr/bin/env python3
"""Interactive GUI for bidirectional scan-line (phase) alignment of TIFF stacks.

Corrects horizontal misalignment between rows scanned in opposite directions
by shifting even rows (0-based) by an integer pixel offset on TIFF time stacks
(frames × Y × X).

Load ChanA and ChanB TIFF stacks directly. They may live in different folders
or processing stages; ChanB is chosen after ChanA with a best-effort path guess.

Preview can use a chosen stack frame or a Z-average for tuning the offset.
Export always applies the offset to every frame individually, writing
<stem>_phase.tif beside each source file."""

from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np
import tifffile

PHASE_SUFFIX = "_phase"
LOG_FILE_NAME = "log.txt"
DEFAULT_OFFSET_RANGE = 40
STACK_FILETYPES = [
    ("TIFF stacks", "*.tif *.tiff *.TIF *.TIFF"),
    ("All files", "*.*"),
]
DISPLAY_MODES = ("Even–Odd", "Turbo", "Grey")
DEFAULT_DISPLAY_MODE = "Even–Odd"
PREVIEW_SOURCES = ("Frame", "Average")
DEFAULT_PREVIEW_SOURCE = "Frame"
PERCENTILE_LOW = 1.0
PERCENTILE_HIGH = 99.0


def load_tif_stack(path: str | Path) -> np.ndarray:
    """Load a TIFF stack as (frames, height, width)."""
    with tifffile.TiffFile(path) as tif:
        stack = tif.asarray()

    if stack.ndim == 2:
        stack = stack[np.newaxis, ...]
    elif stack.ndim != 3:
        raise ValueError(f"Expected 2D or 3D stack, got shape {stack.shape}")

    return stack


def percentile_normalize(
    image: np.ndarray,
    low: float = PERCENTILE_LOW,
    high: float = PERCENTILE_HIGH,
) -> np.ndarray:
    """Stretch image intensities to [0, 1] using percentile clipping."""
    values = np.asarray(image, dtype=np.float64)
    lo, hi = np.percentile(values, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(values)) if values.size else 0.0
        hi = float(np.nanmax(values)) if values.size else 1.0
        if hi <= lo:
            hi = lo + 1.0
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def even_odd_false_color(image: np.ndarray) -> np.ndarray:
    """Cyan (odd rows) / magenta (even rows) overlay after percentile stretch."""
    norm = percentile_normalize(image)
    rgb = np.zeros((*norm.shape, 3), dtype=np.float64)
    # Odd rows: cyan (G + B)
    rgb[1::2, :, 1] = norm[1::2]
    rgb[1::2, :, 2] = norm[1::2]
    # Even rows: magenta (R + B)
    rgb[0::2, :, 0] = norm[0::2]
    rgb[0::2, :, 2] = norm[0::2]
    return rgb


def prepare_preview_image(image: np.ndarray, mode: str) -> tuple[np.ndarray, str | None]:
    """Return display array and matplotlib cmap for the selected preview mode."""
    if mode == "Even–Odd":
        return even_odd_false_color(image), None
    stretched = percentile_normalize(image)
    if mode == "Turbo":
        return stretched, "turbo"
    return stretched, "gray"


def shift_even_rows(image: np.ndarray, offset: int) -> np.ndarray:
    """Shift even rows of a 2D image by ``offset`` pixels (positive = right)."""
    if offset == 0:
        return np.asarray(image).copy()

    result = np.asarray(image).copy()
    height, width = result.shape[:2]
    even = result[0:height:2].copy()

    if offset > 0:
        shift = min(int(offset), width)
        even[:, shift:] = even[:, : width - shift]
        even[:, :shift] = 0
    else:
        shift = min(int(-offset), width)
        even[:, : width - shift] = even[:, shift:]
        even[:, width - shift :] = 0

    result[0:height:2] = even
    return result


def stack_z_average(stack: np.ndarray) -> np.ndarray:
    """Return the mean image across the frame axis."""
    return np.mean(stack, axis=0)


def apply_even_row_shift_to_stack(stack: np.ndarray, offset: int) -> np.ndarray:
    """Apply even-row shift to every frame of a (T, Y, X) stack."""
    if offset == 0:
        return np.asarray(stack).copy()

    corrected = np.empty_like(stack)
    for frame_index, frame in enumerate(stack):
        corrected[frame_index] = shift_even_rows(frame, offset)
    return corrected


def phase_export_path(source_path: Path) -> Path:
    return source_path.with_name(f"{source_path.stem}{PHASE_SUFFIX}{source_path.suffix}")


def write_phase_log(
    folder: Path,
    *,
    offset: int,
    source_path: Path,
    export_path: Path,
    tuning_source: str,
    reference_frame: int | None,
) -> Path:
    """Write log.txt next to the phase TIFF describing the applied offset."""
    log_path = Path(folder) / LOG_FILE_NAME
    lines = [
        "Phase Aligner export log",
        f"source: {source_path.name}",
        f"export: {export_path.name}",
        f"offset_px: {int(offset)}",
        f"tuning_source: {tuning_source}",
    ]
    if tuning_source == "Frame" and reference_frame is not None:
        lines.append(f"reference_frame: {int(reference_frame)}")
    lines.extend(
        [
            "shift_rows: even (0-based)",
            "offset_sign: positive = right",
            "applied_to: every frame individually",
        ]
    )
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log_path


def suggest_chan_b_path(chan_a_path: Path) -> Path | None:
    """Guess a ChanB stack path from a selected ChanA path."""
    path = chan_a_path.resolve()
    path_str = str(path)
    replacements = (
        ("SUPPORT_ChanA", "SUPPORT_ChanB"),
        ("Support_ChanA", "Support_ChanB"),
        ("ChanA", "ChanB"),
        ("chanA", "chanB"),
        ("chana", "chanb"),
        ("CHAN_A", "CHAN_B"),
        ("channel_a", "channel_b"),
        ("ChannelA", "ChannelB"),
    )
    for old, new in replacements:
        if old in path_str:
            candidate = Path(path_str.replace(old, new))
            if candidate.is_file():
                return candidate.resolve()
    return None


def pick_stack_file(
    title: str,
    *,
    initial_dir: Path | None = None,
    initial_file: Path | None = None,
) -> Path | None:
    """Open a file picker for a TIFF stack and return the chosen path."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    kwargs: dict[str, object] = {"title": title, "filetypes": STACK_FILETYPES}
    if initial_dir is not None:
        kwargs["initialdir"] = str(initial_dir)
    if initial_file is not None:
        kwargs["initialfile"] = initial_file.name
    selected = filedialog.askopenfilename(**kwargs)
    root.destroy()
    if not selected:
        return None
    return Path(selected).resolve()


class PhaseAlignerApp:
    def __init__(
        self,
        initial_chan_a: str | Path | None = None,
        initial_chan_b: str | Path | None = None,
    ) -> None:
        self.chan_a_path: Path | None = None
        self.chan_b_path: Path | None = None
        self.stack_a: np.ndarray | None = None
        self.stack_b: np.ndarray | None = None
        self.offset = 0
        self.frame_index = 0
        self.n_frames = 1
        self.display_mode = DEFAULT_DISPLAY_MODE
        self.preview_source = DEFAULT_PREVIEW_SOURCE
        self._updating_frame_slider = False

        self.fig = plt.figure(figsize=(14, 7.8))
        self.fig.canvas.manager.set_window_title("Phase Aligner")

        gs = self.fig.add_gridspec(
            1,
            2,
            left=0.05,
            right=0.98,
            top=0.78,
            bottom=0.18,
            wspace=0.12,
        )
        self.ax_a = self.fig.add_subplot(gs[0, 0])
        self.ax_b = self.fig.add_subplot(gs[0, 1])
        self.image_a = None
        self.image_b = None
        self._preview_is_rgb = False

        self._build_controls()
        self._show_empty_panels()

        if initial_chan_a is not None:
            self.load_chan_a(initial_chan_a, prompt_for_chan_b=initial_chan_b is None)
        if initial_chan_b is not None:
            self.load_chan_b(initial_chan_b)

    def _build_controls(self) -> None:
        ax_preview = self.fig.add_axes([0.05, 0.845, 0.11, 0.095])
        ax_preview.set_title("Tune on", fontsize=8, pad=2)
        self.radio_preview = widgets.RadioButtons(
            ax_preview,
            PREVIEW_SOURCES,
            active=PREVIEW_SOURCES.index(DEFAULT_PREVIEW_SOURCE),
        )
        self.radio_preview.on_clicked(self._on_preview_source_changed)

        ax_mode = self.fig.add_axes([0.185, 0.845, 0.11, 0.095])
        ax_mode.set_title("Display", fontsize=8, pad=2)
        self.radio_mode = widgets.RadioButtons(
            ax_mode,
            DISPLAY_MODES,
            active=DISPLAY_MODES.index(DEFAULT_DISPLAY_MODE),
        )
        self.radio_mode.on_clicked(self._on_display_mode_changed)

        ax_frame = self.fig.add_axes([0.36, 0.910, 0.36, 0.028])
        self.slider_frame = widgets.Slider(
            ax_frame,
            "Frame",
            0,
            1,
            valinit=0,
            valstep=1,
        )
        self.slider_frame.on_changed(self._on_frame_changed)

        ax_offset = self.fig.add_axes([0.36, 0.860, 0.36, 0.028])
        self.slider_offset = widgets.Slider(
            ax_offset,
            "Offset",
            -DEFAULT_OFFSET_RANGE,
            DEFAULT_OFFSET_RANGE,
            valinit=0,
            valstep=1,
        )
        self.slider_offset.on_changed(self._on_offset_changed)

        ax_export = self.fig.add_axes([0.78, 0.875, 0.16, 0.045])
        self.btn_export = widgets.Button(ax_export, "Apply / Export")
        self.btn_export.on_clicked(self._on_export)

        ax_chan_a = self.fig.add_axes([0.05, 0.055, 0.10, 0.045])
        self.btn_chan_a = widgets.Button(ax_chan_a, "ChanA…")
        self.btn_chan_a.on_clicked(self._on_pick_chan_a)

        ax_chan_b = self.fig.add_axes([0.16, 0.055, 0.10, 0.045])
        self.btn_chan_b = widgets.Button(ax_chan_b, "ChanB…")
        self.btn_chan_b.on_clicked(self._on_pick_chan_b)

        self.status_text = self.fig.text(
            0.28,
            0.075,
            "Load ChanA, then ChanB",
            fontsize=9,
            va="center",
            ha="left",
            wrap=True,
        )
    def _show_empty_panels(self) -> None:
        self.ax_a.clear()
        self.ax_b.clear()
        self.ax_a.set_title("ChanA frame")
        self.ax_b.set_title("ChanB frame")
        self.ax_a.set_xticks([])
        self.ax_a.set_yticks([])
        self.ax_b.set_xticks([])
        self.ax_b.set_yticks([])
        self.image_a = None
        self.image_b = None
        self._preview_is_rgb = False
        self.fig.canvas.draw_idle()

    def _set_status(self, message: str) -> None:
        self.status_text.set_text(message)
        self.fig.canvas.draw_idle()

    def _refresh_status(self) -> None:
        if self.chan_a_path is None and self.chan_b_path is None:
            self._set_status("Load ChanA, then ChanB")
            return

        parts: list[str] = []
        if self.chan_a_path is not None:
            shape_a = self.stack_a.shape if self.stack_a is not None else "?"
            parts.append(f"A: {self.chan_a_path}  {shape_a}")
        else:
            parts.append("A: not loaded")

        if self.chan_b_path is not None:
            shape_b = self.stack_b.shape if self.stack_b is not None else "?"
            parts.append(f"B: {self.chan_b_path}  {shape_b}")
        else:
            parts.append("B: not loaded")

        if self.stack_a is not None and self.stack_b is not None:
            tune_note = (
                f"frame={self.frame_index}/{self.n_frames - 1}"
                if self.preview_source == "Frame"
                else "Z-average"
            )
            parts.append(f"tune={tune_note}  |  offset={self.offset} px (even rows)")

        self._set_status("\n".join(parts))
    def _on_preview_source_changed(self, label: str) -> None:
        self.preview_source = label
        use_frame = label == "Frame"
        self.slider_frame.ax.set_visible(use_frame)
        self.slider_frame.label.set_visible(use_frame)
        self._refresh_status()
        self._update_previews()

    def _on_display_mode_changed(self, label: str) -> None:
        self.display_mode = label
        self._update_previews()

    def _on_pick_chan_a(self, _event) -> None:
        initial_dir = self.chan_a_path.parent if self.chan_a_path else None
        path = pick_stack_file("Select ChanA TIFF stack", initial_dir=initial_dir)
        if path is not None:
            self.load_chan_a(path, prompt_for_chan_b=True)

    def _on_pick_chan_b(self, _event) -> None:
        if self.chan_a_path is None:
            messagebox.showinfo(
                "Phase Aligner",
                "Select the ChanA stack first.",
            )
            return
        self._prompt_for_chan_b()

    def _prompt_for_chan_b(self) -> None:
        suggested = suggest_chan_b_path(self.chan_a_path) if self.chan_a_path else None
        if suggested is not None and messagebox.askyesno(
            "Phase Aligner",
            f"Use suggested ChanB stack?\n\n{suggested}",
        ):
            self.load_chan_b(suggested)
            return

        initial_dir = self.chan_a_path.parent if self.chan_a_path else None
        path = pick_stack_file(
            "Select ChanB TIFF stack",
            initial_dir=initial_dir,
            initial_file=suggested,
        )
        if path is not None:
            self.load_chan_b(path)

    def _configure_frame_slider(self, n_frames: int) -> None:
        self.n_frames = max(1, int(n_frames))
        self.frame_index = min(self.frame_index, self.n_frames - 1)
        self._updating_frame_slider = True
        try:
            self.slider_frame.valmin = 0
            self.slider_frame.valmax = self.n_frames - 1
            self.slider_frame.ax.set_xlim(0, max(self.n_frames - 1, 1))
            self.slider_frame.set_val(self.frame_index)
        finally:
            self._updating_frame_slider = False

    def load_chan_a(
        self,
        path: str | Path,
        *,
        prompt_for_chan_b: bool = False,
    ) -> None:
        chan_a = Path(path).resolve()
        if not chan_a.is_file():
            messagebox.showerror("Phase Aligner", f"ChanA stack not found:\n{chan_a}")
            return

        try:
            self._set_status(f"Loading ChanA…\n{chan_a}")
            self.fig.canvas.flush_events()
            stack_a = load_tif_stack(chan_a)
        except Exception as exc:
            messagebox.showerror("Phase Aligner", f"Failed to load ChanA stack:\n{exc}")
            return

        self.chan_a_path = chan_a
        self.stack_a = stack_a
        if self.stack_b is None:
            self.offset = 0
            self.frame_index = 0
            self.slider_offset.set_val(0)
            self._configure_frame_slider(stack_a.shape[0])

        self._refresh_status()
        if self.stack_b is not None:
            self._update_previews()
        else:
            self._show_empty_panels()

        if prompt_for_chan_b:
            self._prompt_for_chan_b()

    def load_chan_b(self, path: str | Path) -> None:
        chan_b = Path(path).resolve()
        if not chan_b.is_file():
            messagebox.showerror("Phase Aligner", f"ChanB stack not found:\n{chan_b}")
            return

        try:
            self._set_status(f"Loading ChanB…\n{chan_b}")
            self.fig.canvas.flush_events()
            stack_b = load_tif_stack(chan_b)
        except Exception as exc:
            messagebox.showerror("Phase Aligner", f"Failed to load ChanB stack:\n{exc}")
            return

        if self.stack_a is not None and stack_b.shape[0] != self.stack_a.shape[0]:
            messagebox.showwarning(
                "Phase Aligner",
                "ChanA and ChanB have different frame counts.\n"
                f"A={self.stack_a.shape[0]}, B={stack_b.shape[0]}.\n"
                "Using the shorter length for the frame slider.",
            )

        self.chan_b_path = chan_b
        self.stack_b = stack_b
        self.offset = 0
        self.frame_index = 0
        self.slider_offset.set_val(0)

        n_frames = stack_b.shape[0]
        if self.stack_a is not None:
            n_frames = min(self.stack_a.shape[0], stack_b.shape[0])
        self._configure_frame_slider(n_frames)

        self._refresh_status()
        self._update_previews()

    def _on_frame_changed(self, value) -> None:
        if self._updating_frame_slider:
            return
        self.frame_index = int(value)
        self._refresh_status()
        self._update_previews()

    def _on_offset_changed(self, value) -> None:
        self.offset = int(value)
        if self.stack_a is None or self.stack_b is None:
            return
        self._refresh_status()
        self._update_previews()

    def _preview_images(self) -> tuple[np.ndarray, np.ndarray, str, str] | None:
        if self.stack_a is None or self.stack_b is None:
            return None
        if self.preview_source == "Average":
            image_a = stack_z_average(self.stack_a)
            image_b = stack_z_average(self.stack_b)
            label_a = "ChanA Z-average"
            label_b = "ChanB Z-average"
        else:
            index = min(
                self.frame_index,
                self.stack_a.shape[0] - 1,
                self.stack_b.shape[0] - 1,
            )
            image_a = self.stack_a[index]
            image_b = self.stack_b[index]
            label_a = f"ChanA frame {index}"
            label_b = f"ChanB frame {index}"
        return image_a, image_b, label_a, label_b

    def _update_previews(self) -> None:
        preview = self._preview_images()
        if preview is None:
            return
        image_a, image_b, label_a, label_b = preview

        preview_a = shift_even_rows(image_a, self.offset)
        preview_b = shift_even_rows(image_b, self.offset)
        display_a, cmap = prepare_preview_image(preview_a, self.display_mode)
        display_b, _ = prepare_preview_image(preview_b, self.display_mode)
        is_rgb = display_a.ndim == 3

        mode_note = {
            "Even–Odd": "odd=cyan, even=magenta",
            "Turbo": "turbo + 1–99% stretch",
            "Grey": "grey + 1–99% stretch",
        }.get(self.display_mode, self.display_mode)

        self.image_a = self._set_panel_image(
            self.ax_a,
            self.image_a,
            display_a,
            cmap=cmap,
            title=f"{label_a} ({mode_note})",
            force_new=is_rgb != self._preview_is_rgb,
        )
        self.image_b = self._set_panel_image(
            self.ax_b,
            self.image_b,
            display_b,
            cmap=cmap,
            title=f"{label_b} ({mode_note})",
            force_new=is_rgb != self._preview_is_rgb,
        )
        self._preview_is_rgb = is_rgb
        self.fig.canvas.draw_idle()

    def _set_panel_image(
        self,
        ax,
        artist,
        data: np.ndarray,
        *,
        cmap: str | None,
        title: str,
        force_new: bool,
    ):
        needs_new = artist is None or force_new
        if needs_new:
            ax.clear()
            if data.ndim == 3:
                artist = ax.imshow(data, interpolation="nearest")
            else:
                artist = ax.imshow(data, cmap=cmap or "gray", interpolation="nearest", vmin=0.0, vmax=1.0)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            return artist

        artist.set_data(data)
        if data.ndim == 2:
            artist.set_clim(0.0, 1.0)
            if cmap is not None:
                artist.set_cmap(cmap)
        ax.set_title(title, fontsize=10)
        return artist

    def _on_export(self, _event) -> None:
        if (
            self.stack_a is None
            or self.stack_b is None
            or self.chan_a_path is None
            or self.chan_b_path is None
        ):
            messagebox.showinfo("Phase Aligner", "Load ChanA and ChanB stacks first.")
            return
        out_a = phase_export_path(self.chan_a_path)
        out_b = phase_export_path(self.chan_b_path)
        if self.preview_source == "Average":
            tune_note = "Z-average preview"
        else:
            tune_note = f"frame {self.frame_index}"
        if not messagebox.askyesno(
            "Apply / Export",
            f"Use offset {self.offset} px tuned on {tune_note}, "
            f"and apply it to every frame of both stacks individually?\n\n"
            f"{out_a}\n{out_b}\n\nContinue?",
        ):
            return

        try:
            self._set_status("Exporting ChanA…")
            self.fig.canvas.flush_events()
            corrected_a = apply_even_row_shift_to_stack(self.stack_a, self.offset)
            tifffile.imwrite(out_a, corrected_a)
            log_a = write_phase_log(
                self.chan_a_path.parent,
                offset=self.offset,
                source_path=self.chan_a_path,
                export_path=out_a,
                tuning_source=self.preview_source,
                reference_frame=self.frame_index if self.preview_source == "Frame" else None,
            )

            self._set_status("Exporting ChanB…")
            self.fig.canvas.flush_events()
            corrected_b = apply_even_row_shift_to_stack(self.stack_b, self.offset)
            tifffile.imwrite(out_b, corrected_b)
            log_b = write_phase_log(
                self.chan_b_path.parent,
                offset=self.offset,
                source_path=self.chan_b_path,
                export_path=out_b,
                tuning_source=self.preview_source,
                reference_frame=self.frame_index if self.preview_source == "Frame" else None,
            )
        except Exception as exc:
            messagebox.showerror("Phase Aligner", f"Export failed:\n{exc}")
            return

        self._set_status(
            f"Exported offset={self.offset} px\n{out_a}\n{out_b}\n{log_a}\n{log_b}"
        )
        messagebox.showinfo(
            "Phase Aligner",
            f"Wrote:\n{out_a}\n{out_b}\n{log_a}\n{log_b}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bidirectional scan-line (phase) aligner for TIFF stacks"
    )
    parser.add_argument(
        "chan_a",
        nargs="?",
        help="Optional path to the ChanA TIFF stack",
    )
    parser.add_argument(
        "chan_b",
        nargs="?",
        help="Optional path to the ChanB TIFF stack",
    )
    args = parser.parse_args()

    initial_a = Path(args.chan_a).resolve() if args.chan_a else None
    initial_b = Path(args.chan_b).resolve() if args.chan_b else None

    PhaseAlignerApp(initial_chan_a=initial_a, initial_chan_b=initial_b)
    plt.show()

if __name__ == "__main__":
    main()
