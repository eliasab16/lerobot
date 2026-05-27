"""Open one camera, request 30 fps MJPEG @ 640x480, and report:
  - what the device accepted (CAP_PROP_FPS, dimensions)
  - the actual measured frame rate every 2 seconds

Press 'q' in the window to quit.
"""

import time

import cv2

CAMERA_INDEX = 0
WIDTH = 640
HEIGHT = 480
TARGET_FPS = 30


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"camera {CAMERA_INDEX}: FAILED to open")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    print(f"camera {CAMERA_INDEX} opened")
    print(f"  requested: {WIDTH}x{HEIGHT} @ {TARGET_FPS} fps MJPEG")
    print(
        f"  accepted:  {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
        f"{cap.get(cv2.CAP_PROP_FPS)} fps"
    )
    print("\nStreaming. Press 'q' to quit.")

    frames = 0
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frames += 1
            cv2.imshow(f"cam{CAMERA_INDEX}", frame)

            now = time.time()
            if now - t0 >= 2.0:
                fps = frames / (now - t0)
                print(f"  measured: {fps:5.1f} fps")
                frames = 0
                t0 = now

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
