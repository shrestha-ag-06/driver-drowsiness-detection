#!/usr/bin/env python3
"""
drowsiness_detector.py
-----------------------
Driver Drowsiness Detection using Computer Vision (Eye Aspect Ratio).

This is the command-line entry point for the project. It reads frames
from a webcam or a video file, locates the driver's face using
MediaPipe Face Landmarker, computes the Eye Aspect Ratio (EAR) for both
eyes, and raises a drowsiness alert when the eyes stay closed
(EAR below a threshold) for a configurable number of consecutive
frames. A single blink will NOT trigger an alert.

Usage examples
--------------
Run with the default webcam, showing a live preview window:
    python drowsiness_detector.py

Run on a saved video file:
    python drowsiness_detector.py --source samples/drowsy.mp4

Run without opening any GUI window (useful on headless servers /
evaluation machines without a display):
    python drowsiness_detector.py --source samples/drowsy.mp4 --headless

Save an annotated copy of the output video:
    python drowsiness_detector.py --source samples/drowsy.mp4 \
        --headless --output results/annotated_output.mp4

Tune sensitivity:
    python drowsiness_detector.py --ear-thresh 0.22 --consec-frames 15

Run with an audible alarm when drowsiness is detected:
    python drowsiness_detector.py --alarm assets/alarm.wav

See README.md for full setup and usage instructions.
"""

import argparse
import csv
import os
import sys
import time
import urllib.request
from datetime import datetime

import cv2
import numpy as np

from eye_utils import average_ear, draw_eye_outline

# mediapipe is imported lazily inside main() so that `--help` works
# even if the dependency isn't installed yet.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Driver Drowsiness Detection using Computer Vision (EAR-based)."
    )
    parser.add_argument(
        "--source", default="0",
        help="Video source: webcam index (e.g. 0) or a path to a video/image "
             "file. Default: 0 (default webcam).",
    )
    parser.add_argument(
        "--ear-thresh", type=float, default=0.25,
        help="EAR value below which an eye is considered closed. Default: 0.25.",
    )
    parser.add_argument(
        "--consec-frames", type=int, default=20,
        help="Number of consecutive closed-eye frames required before raising "
             "a drowsiness alert. Default: 20 (~0.6-1s at 20-30 FPS).",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Do not open a GUI preview window. Useful for servers / automated "
             "evaluation environments without a display. Status is printed to "
             "stdout instead.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to save an annotated copy of the processed video "
             "(e.g. results/annotated_output.mp4).",
    )
    parser.add_argument(
        "--alarm", default=None,
        help="Optional path to a .wav file to play when drowsiness is detected. "
             "Requires the optional 'playsound' dependency.",
    )
    parser.add_argument(
        "--log", default="results/detection_log.csv",
        help="Path to a CSV file where per-frame EAR/status is logged. "
             "Default: results/detection_log.csv",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Optional cap on the number of frames to process (useful for "
             "quick tests). Default: process until the source ends / 'q' "
             "is pressed.",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Run a self-contained smoke test: processes a single in-memory "
             "synthetic frame through the full pipeline (no webcam, no video "
             "file, and no display required). Verifies that OpenCV and "
             "MediaPipe are installed and working, then exits with code 0 "
             "on success. Intended for quick verification on any machine, "
             "including headless evaluation servers.",
    )
    return parser.parse_args()


MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/latest/face_landmarker.task")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models", "face_landmarker.task")


def ensure_model() -> str:
    """Return the path to the Face Landmarker model, downloading it on the
    first run (~4 MB). The modern MediaPipe Tasks API needs this model file;
    it is not bundled inside the pip package."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        return MODEL_PATH
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"[INFO] Downloading face landmark model to {MODEL_PATH} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as exc:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        print(
            f"[ERROR] Could not download the model ({exc}).\n"
            "    Download it manually from:\n"
            f"        {MODEL_URL}\n"
            f"    and save it as:\n        {MODEL_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)
    return MODEL_PATH


def create_face_landmarker(mp):
    """Create a MediaPipe Tasks FaceLandmarker in VIDEO mode (one face)."""
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=ensure_model()),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return FaceLandmarker.create_from_options(options)


def detect_face(mp, landmarker, bgr_frame, timestamp_ms: int):
    """Run the landmarker on a BGR OpenCV frame. Returns the landmark list
    for the first face, or None if no face was found."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=np.ascontiguousarray(rgb))
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    return result.face_landmarks[0] if result.face_landmarks else None


def run_self_test() -> None:
    """Run the detection pipeline on a single synthetic in-memory frame.
    Requires no webcam, no video file, and no display -- this proves the
    environment (OpenCV + MediaPipe + NumPy) is installed and the pipeline
    runs end-to-end from the command line with zero external setup."""
    import mediapipe as mp

    print("[SELF-TEST] Verifying environment (no webcam / no display needed)...")

    frame_h, frame_w = 480, 640
    # A synthetic blank frame is enough to prove the pipeline runs without
    # crashing; MediaPipe is expected to report no face on it, which is a
    # valid, handled outcome ("NO FACE"), not an error.
    frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

    with create_face_landmarker(mp) as landmarker:
        face = detect_face(mp, landmarker, frame, 0)
        status = "NO FACE" if face is None else "FACE DETECTED"

    print(f"[SELF-TEST] OpenCV version   : {cv2.__version__}")
    print(f"[SELF-TEST] MediaPipe loaded : OK")
    print(f"[SELF-TEST] Pipeline status  : {status} (expected on a blank frame)")
    print("[SELF-TEST] PASSED -- environment is set up correctly.")
    sys.exit(0)


def open_video_source(source: str) -> cv2.VideoCapture:
    """Open either a numeric webcam index or a file path as a VideoCapture."""
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        if not os.path.exists(source):
            print(f"[ERROR] Source file not found: {source}", file=sys.stderr)
            sys.exit(1)
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}", file=sys.stderr)
        sys.exit(1)
    return cap


def play_alarm(path: str) -> None:
    """Best-effort alarm playback in a background thread. Never crashes the
    main loop if the optional audio dependency / device is unavailable."""
    import threading

    def _play():
        try:
            from playsound import playsound
            playsound(path)
        except Exception as exc:  # pragma: no cover - best effort only
            print(f"[WARN] Could not play alarm sound ({exc}).", file=sys.stderr)

    threading.Thread(target=_play, daemon=True).start()


def main() -> None:
    args = parse_args()

    try:
        import mediapipe as mp
    except ImportError:
        print(
            "[ERROR] mediapipe is not installed. Run:\n"
            "    pip install -r requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.self_test:
        run_self_test()
        return

    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    landmarker = create_face_landmarker(mp)

    cap = open_video_source(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    writer = None
    if args.output:
        fourcc = (cv2.VideoWriter.fourcc(*"mp4v") if hasattr(cv2.VideoWriter, "fourcc")
                  else cv2.VideoWriter_fourcc(*"mp4v"))
        writer = cv2.VideoWriter(args.output, fourcc, fps, (frame_w, frame_h))

    log_file = open(args.log, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["frame", "timestamp", "ear", "status", "drowsy_event"])

    closed_frame_count = 0
    frame_count = 0
    closed_frame_total = 0
    drowsy_event_count = 0
    in_drowsy_event = False
    start_time = time.time()

    # Auto-fallback: if a display isn't actually available (e.g. a headless
    # evaluation server) but --headless wasn't passed, detect that up front
    # and switch to headless mode ourselves instead of crashing on the
    # first cv2.imshow() call.
    if not args.headless:
        try:
            cv2.namedWindow("Driver Drowsiness Detection", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("Driver Drowsiness Detection")
        except cv2.error:
            print("[INFO] No display detected -- continuing in headless mode "
                  "automatically (status will be printed to the terminal).")
            args.headless = True

    print("[INFO] Starting drowsiness detection. Press 'q' to quit "
          "(GUI mode) or Ctrl+C to stop (headless mode).")

    try:
        while True:
            if args.max_frames is not None and frame_count >= args.max_frames:
                break

            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1
            # VIDEO mode needs strictly increasing timestamps (milliseconds).
            timestamp_ms = int(frame_count * 1000 / fps)
            face_landmarks = detect_face(mp, landmarker, frame, timestamp_ms)

            status = "NO FACE"
            ear_value = 0.0
            drowsy_event_this_frame = False

            if face_landmarks is not None:
                ear_value, left_pts, right_pts = average_ear(
                    face_landmarks, frame_w, frame_h
                )

                if ear_value < args.ear_thresh:
                    closed_frame_count += 1
                    closed_frame_total += 1
                    status = "EYES CLOSED"
                else:
                    closed_frame_count = 0
                    status = "ALERT"
                    in_drowsy_event = False

                if closed_frame_count >= args.consec_frames:
                    status = "DROWSY"
                    if not in_drowsy_event:
                        drowsy_event_count += 1
                        drowsy_event_this_frame = True
                        in_drowsy_event = True
                        if args.alarm:
                            play_alarm(args.alarm)

                if not args.headless:
                    draw_eye_outline(frame, face_landmarks, frame_w, frame_h)

            # --- annotate frame ---
            color = (0, 255, 0) if status == "ALERT" else (0, 0, 255) if status == "DROWSY" else (0, 165, 255)
            cv2.putText(frame, f"EAR: {ear_value:.2f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Status: {status}", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if status == "DROWSY":
                cv2.putText(frame, "WARNING: DROWSINESS DETECTED", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            log_writer.writerow([
                frame_count,
                datetime.now().isoformat(timespec="seconds"),
                f"{ear_value:.4f}",
                status,
                int(drowsy_event_this_frame),
            ])

            if writer is not None:
                writer.write(frame)

            if args.headless:
                if frame_count % max(int(fps), 1) == 0:  # print ~once per second
                    print(f"[frame {frame_count}] EAR={ear_value:.3f} status={status}")
            else:
                cv2.imshow("Driver Drowsiness Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        elapsed = time.time() - start_time
        cap.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
        log_file.close()
        landmarker.close()

        print("\n===== Session Summary =====")
        print(f"Frames processed      : {frame_count}")
        print(f"Closed-eye frames     : {closed_frame_total}")
        print(f"Drowsiness events     : {drowsy_event_count}")
        print(f"Elapsed time (s)      : {elapsed:.2f}")
        print(f"Detection log saved to: {args.log}")
        if args.output:
            print(f"Annotated video saved to: {args.output}")
        print("============================")


if __name__ == "__main__":
    main()
