"""Preview cameras 0, 1, 2 to sanity-check index mapping and frame delivery.

Flags:
    --rotations  CW degrees per camera (0, 90, 180, 270). Default "0,90,90".
    --square     crop | resize | both. Default: no square processing.

Examples:
    python tools/check_cameras.py
    python tools/check_cameras.py --rotations=0,90,270
    python tools/check_cameras.py --square=crop
    python tools/check_cameras.py --square=both
    python tools/check_cameras.py --rotations=0,90,90 --square=crop

Press 'q' (with any window focused) to quit.
"""

import argparse
import threading
import time

import cv2

CAMERA_INDICES = [0, 1, 2]
WIDTH = 640
HEIGHT = 480
TARGET_FPS = 30
USE_MJPEG = True

_ROT_MAP = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def parse_rotations(arg: str) -> dict[int, int | None]:
    parts = [p.strip() for p in arg.split(",") if p.strip()]
    if len(parts) != len(CAMERA_INDICES):
        raise ValueError(
            f"--rotations expects {len(CAMERA_INDICES)} comma-separated values, got {len(parts)}"
        )
    result = {}
    for idx, p in zip(CAMERA_INDICES, parts):
        deg = int(p)
        if deg not in _ROT_MAP:
            raise ValueError(f"Invalid rotation '{deg}' for cam{idx}. Use one of {sorted(_ROT_MAP)}.")
        result[idx] = _ROT_MAP[deg]
    return result


class CameraThread(threading.Thread):
    def __init__(self, idx: int):
        super().__init__(daemon=True)
        self.idx = idx
        self.cap = cv2.VideoCapture(idx)
        self.frame = None
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.frame_count = 0
        self.opened = self.cap.isOpened()

        if self.opened:
            if USE_MJPEG:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
            accepted = (
                int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                self.cap.get(cv2.CAP_PROP_FPS),
            )
            print(f"camera {idx}: opened, accepted {accepted[0]}x{accepted[1]} @ {accepted[2]:.1f} fps")
        else:
            print(f"camera {idx}: FAILED to open")

    def run(self):
        while not self.stop_flag.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                continue
            with self.lock:
                self.frame = frame
                self.frame_count += 1

    def snapshot(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def pop_count(self) -> int:
        with self.lock:
            c = self.frame_count
            self.frame_count = 0
            return c

    def stop(self):
        self.stop_flag.set()
        self.cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rotations",
        default="90,90,90",
        help="CW degrees per camera (comma-separated, one per index). E.g. 90,270,0",
    )
    ap.add_argument(
        "--square",
        choices=["crop", "resize", "both", "off"],
        default="crop",
        help="Make each preview square. 'crop' = center-crop to shorter side. "
             "'resize' = scale to side x side (distorts aspect). "
             "'both' = show crop and resize side by side for comparison. "
             "'off' = native rotated dimensions. Default: crop.",
    )
    args = ap.parse_args()
    rotation = parse_rotations(args.rotations)

    threads = [CameraThread(i) for i in CAMERA_INDICES]
    threads = [t for t in threads if t.opened]
    if not threads:
        print("No cameras opened. Exiting.")
        return

    for t in threads:
        t.start()

    print("\nStreaming. Press 'q' in any window to quit.")
    t_last = time.time()

    try:
        while True:
            for t in threads:
                frame = t.snapshot()
                if frame is None:
                    continue
                rot = rotation.get(t.idx)
                if rot is not None:
                    frame = cv2.rotate(frame, rot)

                if args.square == "off":
                    out = frame
                    cv2.putText(out, f"cam{t.idx}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                else:
                    h, w = frame.shape[:2]
                    side = min(h, w)
                    y0 = (h - side) // 2
                    x0 = (w - side) // 2
                    cropped = frame[y0:y0 + side, x0:x0 + side]
                    resized = cv2.resize(frame, (side, side), interpolation=cv2.INTER_AREA)

                    if args.square == "crop":
                        out = cropped
                        cv2.putText(out, f"cam{t.idx} crop", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    elif args.square == "resize":
                        out = resized
                        cv2.putText(out, f"cam{t.idx} resize", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    else:  # both
                        cv2.putText(cropped, "crop", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                        cv2.putText(resized, "resize", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                        out = cv2.hconcat([cropped, resized])
                        cv2.putText(out, f"cam{t.idx}", (10, out.shape[0] - 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow(f"cam{t.idx}", out)

            now = time.time()
            if now - t_last >= 2.0:
                elapsed = now - t_last
                rates = [(t.idx, t.pop_count() / elapsed) for t in threads]
                msg = "  ".join(f"cam{i}: {r:5.1f} fps" for i, r in rates)
                dead = [f"cam{i}" for i, r in rates if r < 0.5]
                if dead:
                    msg += f"    DEAD: {', '.join(dead)}"
                print(msg)
                t_last = now

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        for t in threads:
            t.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
