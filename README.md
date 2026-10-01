# Driver Drowsiness Detection Using Computer Vision

A real-time, rule-based computer vision system that monitors a driver's
eyes through a webcam (or a recorded video) and raises a **drowsiness
alert** when the eyes stay closed for too long. It works by detecting
facial landmarks around each eye and computing the **Eye Aspect Ratio
(EAR)**, a well-established geometric measure introduced by
Soukupova & Cech (2016) for eye-blink and eye-closure detection.

This is a **traditional / classical computer vision** project — no
custom deep-learning model needs to be trained. Face and eye landmarks
are located using Google's pretrained MediaPipe Face Mesh model, and
drowsiness is decided with simple geometry and temporal (frame-count)
rules on top of that.

---

## 1. How it works

```
Video frame
    |
Face detection + 468 facial landmarks (MediaPipe Face Mesh)
    |
Extract 6 landmark points around each eye
    |
Compute Eye Aspect Ratio (EAR) for left and right eye, average them
    |
EAR < threshold ?  --No-->  Status: ALERT
    |
   Yes
    |
Increment "closed-eye frame" counter
    |
Counter >= N consecutive frames ?  --No-->  Status: EYES CLOSED (still alert)
    |
   Yes
    |
Status: DROWSY -> on-screen warning (+ optional audio alarm)
```

**Eye Aspect Ratio (EAR):**

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```

where `p1..p6` are six points traced around one eye (left corner, two
upper-lid points, right corner, two lower-lid points). EAR stays
roughly constant while an eye is open and drops sharply toward zero
when the eye closes, because the vertical (lid-to-lid) distances
collapse while the horizontal (corner-to-corner) distance barely
changes.

Requiring several **consecutive** closed-eye frames (not just one)
before declaring "drowsy" is what prevents a normal blink from
triggering a false alarm.

---

## 2. Project structure

```
driver-drowsiness-detection/
│
├── README.md               <- this file
├── requirements.txt        <- Python dependencies
├── .gitignore
│
├── drowsiness_detector.py  <- main command-line program
├── eye_utils.py             <- EAR math + landmark helper functions
│
├── samples/                 <- put your own test images/videos here
│   └── README.md
│
├── results/                  <- detection logs / annotated output videos
│   └── .gitkeep
│
└── report/
    └── project_report.md    <- structured project report
```

---

## 3. Requirements

- Python 3.9 – 3.11 (MediaPipe does not yet support every very-recent
  Python version — 3.9/3.10/3.11 are safest)
- A webcam (optional — you can also run on a video file)
- OS: Windows, macOS, or Linux

---

## 4. Setup instructions

Assume you have nothing installed except Python and `git`.

### Step 1 — Clone the repository

```bash
git clone https://github.com/{github-username}/driver-drowsiness-detection.git
cd driver-drowsiness-detection
```

### Step 2 — Create and activate a virtual environment (recommended)

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> If `playsound` fails to install on your platform, that's fine — it is
> only needed for the optional `--alarm` sound feature. Everything else
> still works.

### Step 4 — Run the program

Live webcam demo, with a preview window:
```bash
python drowsiness_detector.py
```
Press `q` in the preview window to stop.

Run on a video file instead of a webcam:
```bash
python drowsiness_detector.py --source samples/drowsy.mp4
```

Run **without any GUI window** (headless — useful for servers, CI, or
automated evaluation machines with no display):
```bash
python drowsiness_detector.py --source samples/drowsy.mp4 --headless
```

Save an annotated copy of the processed video:
```bash
python drowsiness_detector.py --source samples/drowsy.mp4 --headless \
    --output results/drowsy_annotated.mp4
```

Play a sound when drowsiness is detected:
```bash
python drowsiness_detector.py --alarm assets/alarm.wav
```

Tune sensitivity (lower threshold / fewer frames = more sensitive):
```bash
python drowsiness_detector.py --ear-thresh 0.22 --consec-frames 15
```

See all options:
```bash
python drowsiness_detector.py --help
```

### Command-line options

| Flag | Default | Description |
|---|---|---|
| `--source` | `0` | Webcam index, or path to a video/image file |
| `--ear-thresh` | `0.25` | EAR value below which an eye counts as closed |
| `--consec-frames` | `20` | Consecutive closed-eye frames required to trigger DROWSY |
| `--headless` | off | Run without opening a GUI preview window |
| `--output` | none | Path to save an annotated output video |
| `--alarm` | none | Path to a `.wav` file to play on a drowsiness event |
| `--log` | `results/detection_log.csv` | Path to the per-frame CSV log |
| `--max-frames` | none | Stop after N frames (useful for quick tests) |

---

## 5. Output

For every run, the program:

1. Prints a live status line per frame (headless mode) or overlays
   `EAR`, `Status`, and a `WARNING: DROWSINESS DETECTED` banner
   directly on the video (GUI mode).
2. Writes a CSV log to `results/detection_log.csv` with one row per
   frame: `frame, timestamp, ear, status, drowsy_event`.
3. Prints a session summary on exit: total frames processed,
   closed-eye frame count, number of distinct drowsiness events, and
   elapsed time.

Example summary:
```
===== Session Summary =====
Frames processed      : 452
Closed-eye frames     : 61
Drowsiness events     : 2
Elapsed time (s)      : 15.03
Detection log saved to: results/detection_log.csv
============================
```

---

## 6. Quick test without a webcam

If you don't have a webcam or sample video handy, you can still verify
the program runs correctly:

```bash
python drowsiness_detector.py --source 0 --headless --max-frames 30
```

This will attempt to open the default webcam for 30 frames and exit
cleanly, confirming the environment is set up correctly. If no webcam
is available (e.g. on a CI runner), supply `--source` with a path to
any short video file instead.

---

## 7. Limitations

- Detection accuracy depends on lighting; very dark or backlit scenes
  reduce landmark quality.
- Heavy face occlusion (masks, hands near the face) can prevent
  face-mesh detection entirely.
- Glasses (especially reflective or dark lenses) can occasionally
  interfere with eye-landmark accuracy.
- The fixed EAR threshold is not calibrated per individual; eye shape
  varies between people, so the default `0.25` may need tuning.
- This is a prototype for educational purposes and is **not** a
  safety-certified driver-monitoring system.

---

## 8. Technologies used

- **Python 3** — main language
- **OpenCV** — video I/O, drawing, display
- **MediaPipe Face Mesh** — pretrained facial landmark detection (468
  landmarks per face)
- **NumPy** — vector/distance math for the EAR calculation

---

## 9. Report

See [`report/project_report.md`](report/project_report.md) for the full
structured project report (problem statement, methodology, results,
evaluation, and limitations).
