"""
camera_utils.py - Opens the webcam reliably on Mac / Windows / Linux.

Test your camera on its own:
    python camera_utils.py            # finds every camera and shows a preview
    python camera_utils.py --camera 1 # preview a specific camera
"""
import argparse
import platform
import sys
import time

import cv2

SYSTEM = platform.system()

HELP = """
Could not get video from the camera. Try these, in order:

 1. PERMISSION (most common on Mac)
    System Settings > Privacy & Security > Camera
    -> turn ON the app you run Python from (Terminal, iTerm, or Visual Studio Code).
    Then QUIT that app completely (Cmd+Q) and open it again.
    If the app is not in the list, reset the permission so macOS asks again:
        tccutil reset Camera com.apple.Terminal        (for Terminal)
        tccutil reset Camera com.microsoft.VSCode      (for VS Code)
    and run the script again - click "OK" on the popup.

 2. CAMERA BUSY - close Zoom, Google Meet, FaceTime, Photo Booth, browser tabs using the camera.

 3. WRONG CAMERA - an iPhone (Continuity Camera) may be camera 0. Try:
        python capture_dataset.py --camera 1

 4. Run  python camera_utils.py  to list which camera numbers work.
"""


def _backend():
    if SYSTEM == "Darwin":
        return cv2.CAP_AVFOUNDATION      # native Mac camera API
    if SYSTEM == "Windows":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def _try_open(index, warmup_seconds=3.0):
    """Open one camera index and wait until it gives a real frame."""
    cap = cv2.VideoCapture(index, _backend())
    if not cap.isOpened():
        cap.release()
        return None
    deadline = time.time() + warmup_seconds
    while time.time() < deadline:            # Mac cameras need a moment to start
        ok, frame = cap.read()
        if ok and frame is not None and frame.size:
            return cap, frame
        time.sleep(0.1)
    cap.release()
    return None


def open_camera(index=None, max_index=3):
    """Returns an opened cv2.VideoCapture, or exits with clear instructions."""
    indices = [index] if index is not None else range(max_index)
    for i in indices:
        result = _try_open(i)
        if result:
            cap, frame = result
            print(f"[camera] using camera {i}  ({frame.shape[1]}x{frame.shape[0]})")
            if frame.mean() < 2:
                print("[camera] WARNING: picture is completely black - check the lens / "
                      "privacy shutter, or camera permission (see below).\n" + HELP)
            return cap
        print(f"[camera] camera {i}: no video")
    print(HELP)
    sys.exit(1)


def read_frame(cap, retries=30):
    """Read a frame, tolerating a few dropped frames instead of quitting."""
    for _ in range(retries):
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
        time.sleep(0.03)
    return None


def main():
    ap = argparse.ArgumentParser(description="Camera test")
    ap.add_argument("--camera", type=int, default=None)
    args = ap.parse_args()

    if args.camera is None:
        print("Searching for cameras...")
        found = []
        for i in range(4):
            result = _try_open(i, 2.0)
            if result:
                result[0].release()
                found.append(i)
        print(f"Working cameras: {found if found else 'none'}")
        if not found:
            print(HELP)
            sys.exit(1)
    cap = open_camera(args.camera)
    print("Preview open - press q in the video window to quit.")
    while True:
        frame = read_frame(cap)
        if frame is None:
            print("Camera stopped sending frames.")
            break
        cv2.imshow("Camera test (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
