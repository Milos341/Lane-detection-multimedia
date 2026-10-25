from collections import deque
from pathlib import Path
import sys

import cv2
import numpy as np

from calibration import undistort
from thresholding import binary_threshold, IMG_SHAPE
from perspective import warp
from lane_finding import find_lane_pixels
from measurements import measure_lane
from pipeline import MTX, DIST, M, MINV, draw_lane

N_AVG = 10            # koliko poslednjih prihvacenih kadrova se usrednjava
MIN_PIXELS = 100      # minimalan broj piksela po liniji da bi se fit uopste racunao
WIDTH_RANGE = (450, 800)   # dozvoljena sirina trake u pikselima (~2.6 do 4.6 m)
MAX_WIDTH_DIFF = 150       # najvece dozvoljeno odstupanje sirine izmedju vrha i dna


class LaneTracker:
    """Pamti poslednje ispravne fitove i usrednjava ih izmedju kadrova."""

    def __init__(self, n=N_AVG):
        self.left_fits = deque(maxlen=n)
        self.right_fits = deque(maxlen=n)
        self.curvs = deque(maxlen=n)
        self.offsets = deque(maxlen=n)
        self.bad_frames = 0

    @staticmethod
    def is_sane(left_fit, right_fit, height):
        """Provera: razumna sirina trake i priblizna paralelnost linija."""
        ploty = np.array([0, height / 2, height - 1])
        widths = np.polyval(right_fit, ploty) - np.polyval(left_fit, ploty)
        if np.any(widths < WIDTH_RANGE[0]) or np.any(widths > WIDTH_RANGE[1]):
            return False
        return (widths.max() - widths.min()) <= MAX_WIDTH_DIFF

    def update(self, warped):
        h, w = warped.shape
        leftx, lefty, rightx, righty, _ = find_lane_pixels(warped)
        if len(leftx) >= MIN_PIXELS and len(rightx) >= MIN_PIXELS:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            if self.is_sane(left_fit, right_fit, h):
                curv, off, _ = measure_lane(leftx, lefty, rightx, righty, w, h)
                self.left_fits.append(left_fit)
                self.right_fits.append(right_fit)
                self.curvs.append(curv)
                self.offsets.append(off)
                return True
        self.bad_frames += 1
        return False

    def best(self):
        """Usrednjeni fit, ili None ako jos nijedan kadar nije bio ispravan."""
        if not self.left_fits:
            return None
        return (np.mean(self.left_fits, axis=0), np.mean(self.right_fits, axis=0),
                float(np.median(self.curvs)), float(np.mean(self.offsets)))

    def process(self, frame):
        und = undistort(frame, MTX, DIST)
        warped = warp(binary_threshold(und)["combined"], M)
        self.update(warped)

        best = self.best()
        if best is None:
            return und
        left_fit, right_fit, curv, off = best

        h = warped.shape[0]
        ploty = np.linspace(0, h - 1, h)
        left_fitx = np.polyval(left_fit, ploty)
        right_fitx = np.polyval(right_fit, ploty)
        result = draw_lane(und, warped.shape, ploty, left_fitx, right_fitx, MINV)

        curv_text = ("Curvature radius: straight" if curv > 5000
                     else f"Curvature radius: {curv:.0f} m")
        side = "right" if off > 0 else "left"
        cv2.putText(result, curv_text, (40, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 255, 255), 2)
        cv2.putText(result, f"Vehicle is {abs(off):.2f} m {side} of center", (40, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        return result


def process_video(in_path, out_path):
    cap = cv2.VideoCapture(str(in_path))
    w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    if (h, w) != IMG_SHAPE:
        print(f"Preskoceno (druga rezolucija {w}x{h}): {Path(in_path).name}")
        cap.release()
        return

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    tracker = LaneTracker()
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(tracker.process(frame))
        n += 1
        if n % 100 == 0:
            print(f"  {n} kadrova...")
    cap.release()
    writer.release()
    print(f"Gotovo: {out_path} ({n} kadrova, odbacenih fitova: {tracker.bad_frames})")


if __name__ == "__main__":
    inputs = sys.argv[1:] or ["test_videos/project_video03.mp4"]
    out_dir = Path("output_videos")
    out_dir.mkdir(exist_ok=True)
    for p in inputs:
        print(f"Obrada: {p}")
        process_video(p, out_dir / f"{Path(p).stem}_out.mp4")