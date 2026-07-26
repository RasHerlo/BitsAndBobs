# BitsAndBobs

Small analysis tools for imaging workflows. The two main GUIs use **separate Python virtual environments** so their dependencies stay isolated.

| Tool | Script(s) | Environment | Requirements |
|------|-----------|-------------|--------------|
| Image Aligner | `Image_Aligner.py` | `venv_image_aligner` | `requirements_image_aligner.txt` |
| Stack Analyzer | `stack_analyzer.py`, `stack_analyzer_total.py` | `venv_stack_analyzer` | `requirements_stack_analyzer.txt` |

Shared helper used by Stack Analyzer / Total: `portable_paths.py` (drive-flexible path resolution for USB / remounted drives).

---

## Environments

### Image Aligner

```powershell
python -m venv venv_image_aligner
.\venv_image_aligner\Scripts\Activate.ps1
pip install -r requirements_image_aligner.txt
```

Dependencies: `numpy`, `matplotlib`.

### Stack Analyzer (+ Total)

```powershell
python -m venv venv_stack_analyzer
.\venv_stack_analyzer\Scripts\Activate.ps1
pip install -r requirements_stack_analyzer.txt
```

Dependencies: `numpy`, `matplotlib`, `scipy`, `tifffile` (plus stdlib `tkinter`).

Both venvs are gitignored (`venv_image_aligner/`, `venv_stack_analyzer/`).

---

## Image Aligner (`Image_Aligner.py`)

Interactive tool for aligning two images and checking how well they overlap.

**What it does**
- Loads two images and shows them as cyan / magenta
- Interactive overlay with keyboard or on-screen arrow controls
- Adjustable step size for translation
- Local NCC (normalized cross-correlation) map over a central analysis region
- Global NCC score for the current alignment

**Run**

```powershell
.\venv_image_aligner\Scripts\Activate.ps1
python Image_Aligner.py path\to\image1.tif path\to\image2.tif
```

---

## Stack Analyzer (`stack_analyzer.py`)

Interactive GUI for fluorescence ROI analysis on TIFF stacks.

**What it does**
- Load a TIFF stack and view the z-average image
- Draw / edit an ROI and optional background (BG) ROI
- Extract raw fluorescence traces; smooth with Savitzky–Golay
- Bleach correction (biexponential fit) and BC-corrected normalized traces
- Stimulus / event timing via start frames, extension window, acquisition fps, and averaging factor
- Overlay aligned event segments and integrate response area
- Optional pixel-wise heatmap of response area
- Persist quantified ROIs to `ROI_quant pickle.pkl` next to the stack
- **Inspect Pickle** — browse saved ROI quantification rows
- **Mark Events** — inspect saved ROIs, adjust BC baseline shift, add/remove marked event intervals
- Drive-flexible directory matching so ROI rows still match when a USB remounts under a different drive letter

**Run**

```powershell
.\venv_stack_analyzer\Scripts\Activate.ps1
python stack_analyzer.py
# or with an initial stack:
python stack_analyzer.py path\to\stack.tif
```

---

## Stack Analyzer Total (`stack_analyzer_total.py`)

Aggregation and summary GUI that combines ROI quantification pickles from multiple experiments. Uses the **same** `venv_stack_analyzer` environment as Stack Analyzer.

**What it does**
- Choose a **collect directory** and manage `pickle_stack_collect.pkl`
- Add / remove experiment pickles (`ROI_quant pickle.pkl` files or their folders)
- **Rebuild Collection** — re-read all experiment pickles and rebuild collected rows (evoked / marked / non-event segments)
- **Inspect Pickle** — inspect the collection store (rows, directories, segments)
- **Results** — multi-panel summary figure (overlays, timing, peak maxima, durations) with export
- Stores experiment and row paths relative to the collect directory so collections remain usable when the USB drive letter changes (`C:`, `D:`, `F:`, …)

**Run**

```powershell
.\venv_stack_analyzer\Scripts\Activate.ps1
python stack_analyzer_total.py
# or with an initial collect directory:
python stack_analyzer_total.py --collect-dir path\to\collection_folder
```

**Typical workflow**
1. Analyze each stack in `stack_analyzer.py` and save ROIs / mark events.
2. In `stack_analyzer_total.py`, point at a collect folder, add the experiment pickles, and rebuild.
3. Open **Results** (and optionally export) for cross-experiment summaries.
