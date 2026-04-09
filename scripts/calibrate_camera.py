from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-time camera intrinsics calibration using a chessboard.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (use external webcam index).")
    parser.add_argument("--cam-width", type=int, default=1920, help="Requested capture width.")
    parser.add_argument("--cam-height", type=int, default=1080, help="Requested capture height.")
    parser.add_argument("--board-cols", type=int, default=9, help="Chessboard inner corners count along width.")
    parser.add_argument("--board-rows", type=int, default=6, help="Chessboard inner corners count along height.")
    parser.add_argument("--square-mm", type=float, default=20.0, help="Chessboard square size in mm.")
    parser.add_argument("--min-frames", type=int, default=20, help="Minimum valid samples before solve.")
    parser.add_argument("--output", default="configs/camera_intrinsics.json", help="Output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = read_args()
    board_size = (int(args.board_cols), int(args.board_rows))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)

    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)
    objp *= float(args.square_mm)

    objpoints: list[np.ndarray] = []
    imgpoints: list[np.ndarray] = []

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
    cv2.namedWindow("camera-calibration", cv2.WINDOW_AUTOSIZE)

    print("Calibration instructions:")
    print("1) Show printed chessboard from different angles/distances.")
    print("2) Press SPACE only when corners are detected.")
    print("3) Collect at least --min-frames samples, then press ENTER to solve.")
    print("4) Press Q or ESC to cancel.")

    last_gray_shape: tuple[int, int] | None = None
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        view = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray,
            board_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        status = f"Samples: {len(objpoints)} / {args.min_frames}"
        cv2.putText(view, status, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        corner_status = "Corners: FOUND" if found else "Corners: NOT FOUND"
        corner_color = (0, 220, 0) if found else (0, 0, 255)
        cv2.putText(view, corner_status, (14, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, corner_color, 2)
        cv2.putText(
            view,
            f"Pattern expected: {args.board_cols}x{args.board_rows} inner corners",
            (14, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2,
        )
        if found:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(view, board_size, corners2, found)
            last_gray_shape = gray.shape[::-1]
        cv2.imshow("camera-calibration", view)
        key = cv2.waitKey(15) & 0xFF
        if key in (27, ord("q")):
            cap.release()
            cv2.destroyAllWindows()
            return
        if key == ord(" "):
            if found:
                objpoints.append(objp.copy())
                imgpoints.append(corners2.copy())
                print(f"Captured sample {len(objpoints)}")
            else:
                print("No corners detected in current frame. Move board, reduce glare, or fix pattern size.")
        if key in (10, 13):
            if len(objpoints) >= args.min_frames:
                break
            print(f"Need at least {args.min_frames} samples, have {len(objpoints)}")

    if not objpoints or last_gray_shape is None:
        cap.release()
        cv2.destroyAllWindows()
        raise RuntimeError("No valid calibration samples collected.")

    ret, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        last_gray_shape,
        None,
        None,
    )

    payload = {
        "rms_reprojection_error": float(ret),
        "image_size": [int(last_gray_shape[0]), int(last_gray_shape[1])],
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.reshape(-1).tolist(),
        "fx": float(camera_matrix[0, 0]),
        "fy": float(camera_matrix[1, 1]),
        "cx": float(camera_matrix[0, 2]),
        "cy": float(camera_matrix[1, 2]),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved camera intrinsics to: {out}")
    print(f"RMS reprojection error: {ret:.4f} (lower is better, target < 0.5)")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
