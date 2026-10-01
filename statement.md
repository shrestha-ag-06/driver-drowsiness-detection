# Problem Statement

**Project:** Driver Drowsiness Detection Using Computer Vision

---

## 1. Background

Driver fatigue is a major contributor to road accidents. A drowsy driver
reacts more slowly, loses focus, and in the worst case falls asleep at
the wheel, often without any warning. One of the clearest visible signs
of fatigue is **prolonged eye closure**: while a normal blink lasts only a
fraction of a second, a drowsy driver's eyes stay closed for noticeably
longer.

Relying on drivers to judge their own alertness is unreliable, because
fatigue impairs the very judgment needed to recognise it. An automated,
camera-based monitor can watch the driver continuously and raise an alert
before a fatigue-related incident occurs.

## 2. Problem Definition

Design and implement a **real-time computer vision system** that watches
a driver's face through a webcam (or a recorded video) and automatically
decides whether the driver is drowsy, based on how long their eyes remain
closed.

The system must be able to:

1. **Detect** the driver's face and eye regions in every video frame.
2. **Measure** eye openness numerically, frame by frame.
3. **Distinguish** a normal blink from sustained eye closure.
4. **Alert** the driver clearly and promptly when sustained closure
   (drowsiness) is detected.

## 3. Proposed Approach

The solution uses **classical, rule-based computer vision** rather than a
custom-trained deep-learning model:

- **Landmark detection:** MediaPipe Face Mesh (pretrained) locates the
  face and the landmarks around each eye.
- **Eye openness measure:** the **Eye Aspect Ratio (EAR)** (Soukupová &
  Čech, 2016) is computed from six landmark points per eye and averaged
  across both eyes:

  ```
  EAR = (‖p2 − p6‖ + ‖p3 − p5‖) / (2 · ‖p1 − p4‖)
  ```

  EAR stays roughly constant while the eye is open and drops sharply
  toward zero when it closes.
- **Temporal decision rule:** if EAR stays below a threshold for a set
  number of **consecutive** frames (default: 0.25 for 20 frames, roughly
  one second), the driver is classified as **DROWSY**. Shorter dips, such
  as ordinary blinks, are ignored.
- **Feedback and logging:** an on-screen warning (plus an optional audio
  alarm) is shown on detection, and every frame's result is saved to a
  CSV log.

## 4. Inputs and Outputs

| | Description |
|---|---|
| **Input** | Live webcam stream, or a recorded video/image file |
| **Processing** | Face landmarks → EAR per eye → averaged EAR → consecutive-frame check |
| **Output** | Per-frame status (`ALERT` / `EYES CLOSED` / `DROWSY`), on-screen warning banner, optional audio alarm, optional annotated output video, CSV log (`frame, timestamp, ear, status, drowsy_event`), and an end-of-run summary |

## 5. Scope and Constraints

**In scope**
- Eye-closure-based drowsiness detection from a single front-facing camera
- Real-time operation on a standard laptop CPU, no GPU or training required
- Command-line use, with headless mode for machines without a display
- Tunable sensitivity (`--ear-thresh`, `--consec-frames`)

**Out of scope**
- Other fatigue cues (yawning, head nodding, blink-rate trends)
- Per-user calibration
- Night or infrared driving conditions
- Certification as an in-vehicle safety system. This is an educational
  prototype.

## 6. Success Criteria

- A normal blink does **not** trigger a drowsiness alert.
- Sustained eye closure triggers a `DROWSY` alert within about one second.
- The system runs on live webcam input and on recorded video.
- Every run produces a reproducible CSV log and a session summary
  (frames processed, closed-eye frames, drowsiness events, elapsed time).
- The project can be installed and run from the repository using only the
  documented setup steps.

## 7. Known Limitations

- Poor lighting, backlighting, masks, hands over the face, or extreme
  head poses can degrade or break landmark tracking.
- Glasses with glare or dark lenses may distort eye landmarks.
- The fixed EAR threshold is not personalised, so it may need tuning for
  different eye shapes and camera angles.

## 8. Reference

Soukupová, T., & Čech, J. (2016). *Real-Time Eye Blink Detection using
Facial Landmarks.* 21st Computer Vision Winter Workshop.
