import argparse
import csv
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_SPECS = {
    "lite": {
        "url": (
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
        ),
        "path": Path("models/pose_landmarker_lite.task"),
    },
    "full": {
        "url": (
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
        ),
        "path": Path("models/pose_landmarker_full.task"),
    },
    "heavy": {
        "url": (
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
        ),
        "path": Path("models/pose_landmarker_heavy.task"),
    },
}

LEFT_EAR_INDEX = 7
RIGHT_EAR_INDEX = 8


def ensure_pose_model(model_path: Path, model_url: str) -> None:
    if model_path.exists():
        return

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose model to {model_path}...")
    urlretrieve(model_url, model_path)


def create_landmarker(model_path: Path):
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Golf Coach with MediaPipe Tasks")
    parser.add_argument("--video", default="media/01.mp4", help="Input video path")
    parser.add_argument(
        "--model",
        choices=["lite", "full", "heavy"],
        default="full",
        help="Pose model variant",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Custom .task model path (skip preset auto selection)",
    )
    parser.add_argument(
        "--csv",
        default="media/head_metrics.csv",
        help="CSV output path for per-frame metrics",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable OpenCV preview window",
    )
    return parser.parse_args()


def resolve_model(model: str = "full", model_path: str = None) -> tuple[Path, str]:
    if model_path:
        resolved_path = Path(model_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Model file not found: {resolved_path}")
        return resolved_path, "custom"

    spec = MODEL_SPECS[model]
    resolved_path = spec["path"]
    ensure_pose_model(resolved_path, spec["url"])
    return resolved_path, model


def create_video_writer(
    output_video_path: Path,
    fps: float,
    frame_w: int,
    frame_h: int,
) -> tuple[cv2.VideoWriter, str]:
    # Prefer browser-friendly codecs first.
    codec_candidates = ["avc1", "H264", "X264", "mp4v"]
    for codec in codec_candidates:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_w, frame_h))
        if writer.isOpened():
            return writer, codec
        writer.release()

    raise RuntimeError("Failed to initialize video writer with supported codecs")


def process_video(
    input_video_path: str,
    output_video_path: str,
    csv_output_path: str,
    model: str = "full",
    model_path: str = None,
    show_preview: bool = True,
) -> dict:
    resolved_model_path, resolved_model_name = resolve_model(model=model, model_path=model_path)

    input_video = Path(input_video_path)
    output_video = Path(output_video_path)
    csv_output = Path(csv_output_path)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    csv_output.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out, used_codec = create_video_writer(output_video, fps, frame_w, frame_h)

    is_first = True
    first_center_x, first_center_y, first_radius = None, None, None
    last_timestamp_ms = 0
    frame_interval_ms = int(1000 / fps)
    frame_idx = 0
    moving_frames = 0
    max_lateral_deviation_px = 0

    try:
        with open(csv_output, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "frame",
                    "timestamp_ms",
                    "left_ear_visibility",
                    "right_ear_visibility",
                    "center_x",
                    "center_y",
                    "radius",
                    "out_of_reference",
                ]
            )

            with create_landmarker(resolved_model_path) as landmarker:
                while cap.isOpened():
                    ret, img = cap.read()
                    if not ret:
                        break

                    img_h, img_w, _ = img.shape
                    img_result = img.copy()

                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

                    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                    if timestamp_ms <= last_timestamp_ms:
                        timestamp_ms = last_timestamp_ms + frame_interval_ms
                    last_timestamp_ms = timestamp_ms

                    results = landmarker.detect_for_video(mp_image, timestamp_ms)
                    out_of_reference = False
                    center_x, center_y, radius = None, None, None
                    left_vis, right_vis = 0.0, 0.0

                    if results.pose_landmarks:
                        landmark = results.pose_landmarks[0]

                        left_ear_x = landmark[LEFT_EAR_INDEX].x * img_w
                        left_ear_y = landmark[LEFT_EAR_INDEX].y * img_h

                        right_ear_x = landmark[RIGHT_EAR_INDEX].x * img_w
                        right_ear_y = landmark[RIGHT_EAR_INDEX].y * img_h
                        left_vis = float(landmark[LEFT_EAR_INDEX].visibility)
                        right_vis = float(landmark[RIGHT_EAR_INDEX].visibility)

                        center_x = int((left_ear_x + right_ear_x) / 2)
                        center_y = int((left_ear_y + right_ear_y) / 2)

                        radius = int(abs(left_ear_x - right_ear_x) / 2)
                        radius = max(radius, 20)

                        if is_first:
                            first_center_x = center_x
                            first_center_y = center_y
                            first_radius = int(radius * 2)
                            is_first = False
                        else:
                            cv2.circle(
                                img_result,
                                center=(first_center_x, first_center_y),
                                radius=first_radius,
                                color=(0, 255, 255),
                                thickness=2,
                            )

                            color = (0, 255, 0)
                            if (
                                center_x - radius < first_center_x - first_radius
                                or center_x + radius > first_center_x + first_radius
                            ):
                                color = (0, 0, 255)
                                out_of_reference = True

                            cv2.circle(
                                img_result,
                                center=(center_x, center_y),
                                radius=radius,
                                color=color,
                                thickness=2,
                            )

                    overlay_text = (
                        f"Model: {resolved_model_name.upper()}  "
                        f"L_vis: {left_vis:.2f}  R_vis: {right_vis:.2f}"
                    )
                    cv2.putText(
                        img_result,
                        overlay_text,
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    status_text = "Head Stable" if not out_of_reference else "Head Moving"
                    status_color = (0, 255, 0) if not out_of_reference else (0, 0, 255)
                    cv2.putText(
                        img_result,
                        status_text,
                        (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        status_color,
                        2,
                        cv2.LINE_AA,
                    )

                    writer.writerow(
                        [
                            frame_idx,
                            timestamp_ms,
                            round(left_vis, 4),
                            round(right_vis, 4),
                            center_x,
                            center_y,
                            radius,
                            int(out_of_reference),
                        ]
                    )

                    if out_of_reference:
                        moving_frames += 1

                    if not is_first and center_x is not None:
                        lateral_deviation_px = abs(center_x - first_center_x)
                        if lateral_deviation_px > max_lateral_deviation_px:
                            max_lateral_deviation_px = lateral_deviation_px

                    frame_idx += 1

                    if show_preview:
                        cv2.imshow("AI Golf Coach", img_result)
                    out.write(img_result)

                    if show_preview and cv2.waitKey(1) == ord("q"):
                        break
    finally:
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()

    moving_ratio_percent = (moving_frames / frame_idx * 100.0) if frame_idx else 0.0
    return {
        "total_frames": frame_idx,
        "moving_frames": moving_frames,
        "moving_ratio_percent": moving_ratio_percent,
        "max_lateral_deviation_px": max_lateral_deviation_px,
        "output_video_path": str(output_video),
        "csv_output_path": str(csv_output),
        "model": resolved_model_name,
        "video_codec": used_codec,
    }


def main() -> None:
    args = parse_args()
    video_path = args.video
    print(video_path)

    output_video_path = f"{Path(video_path).with_suffix('')}_output.mp4"
    summary = process_video(
        input_video_path=video_path,
        output_video_path=output_video_path,
        csv_output_path=args.csv,
        model=args.model,
        model_path=args.model_path,
        show_preview=not args.no_preview,
    )

    print("\n=== Session Summary ===")
    print(f"Total frames: {summary['total_frames']}")
    print(
        "Head moving frames: "
        f"{summary['moving_frames']} ({summary['moving_ratio_percent']:.2f}%)"
    )
    print(f"Max lateral deviation: {summary['max_lateral_deviation_px']}px")


if __name__ == "__main__":
    main()
