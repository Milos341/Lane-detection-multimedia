from pathlib import Path
import glob

import cv2
import numpy as np

from calibration import undistort
from thresholding import load_calibration, binary_threshold, IMG_SHAPE
from perspective import get_transforms, warp
from lane_finding import find_lane_pixels

# Procena pretvaranja piksela u metre (pogled odozgo)
YM_PER_PIX = 30 / 720   # pretpostavka: ~30 m puta u vidljivom delu
XM_PER_PIX = 3.7 / 640  # standardna traka 3.7 m = 640 px izmedju DST linija


def measure_lane(leftx, lefty, rightx, righty, img_width, img_height):
    """Vraca (poluprecnik_m, pomeraj_m, sirina_trake_m).

    pomeraj_m > 0 znaci da je vozilo desno od centra trake, < 0 levo.
    """
    # Fit u metrima (x = A*y^2 + B*y + C)
    left_fit_m = np.polyfit(lefty * YM_PER_PIX, leftx * XM_PER_PIX, 2)
    right_fit_m = np.polyfit(righty * YM_PER_PIX, rightx * XM_PER_PIX, 2)

    # Poluprecnik racunamo na dnu slike (tacka najbliza vozilu)
    y_eval = (img_height - 1) * YM_PER_PIX

    def radius(fit):
        A, B, _ = fit
        return (1 + (2 * A * y_eval + B) ** 2) ** 1.5 / max(abs(2 * A), 1e-9)

    curvature = (radius(left_fit_m) + radius(right_fit_m)) / 2

    # Pozicije linija na dnu slike, u pikselima
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    y_bottom = img_height - 1
    left_x = np.polyval(left_fit, y_bottom)
    right_x = np.polyval(right_fit, y_bottom)

    lane_center = (left_x + right_x) / 2
    vehicle_center = img_width / 2  # kamera je na sredini vozila
    offset = (vehicle_center - lane_center) * XM_PER_PIX
    lane_width = (right_x - left_x) * XM_PER_PIX
    return curvature, offset, lane_width


if __name__ == "__main__":
    mtx, dist = load_calibration()
    M, Minv = get_transforms()

    for p in sorted(glob.glob("test_images/*.jpg")):
        img = cv2.imread(p)
        if img.shape[:2] != IMG_SHAPE:
            continue
        und = undistort(img, mtx, dist)
        warped = warp(binary_threshold(und)["combined"], M)
        leftx, lefty, rightx, righty, _ = find_lane_pixels(warped)
        if len(leftx) < 3 or len(rightx) < 3:
            print(f"{Path(p).stem}: nije pronadjena linija")
            continue
        h, w = warped.shape
        curv, off, width = measure_lane(leftx, lefty, rightx, righty, w, h)
        side = "desno" if off > 0 else "levo"
        print(f"{Path(p).stem:16s} R = {curv:8.0f} m | "
              f"pomeraj = {abs(off):.2f} m {side} od centra | sirina trake = {width:.2f} m")