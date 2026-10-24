from pathlib import Path
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt

from calibration import undistort
from thresholding import load_calibration, binary_threshold, IMG_SHAPE
from perspective import get_transforms, warp


def find_lane_pixels(binary_warped, nwindows=9, margin=100, minpix=50):
    """Trazi piksele leve i desne linije histogramom i kliznim prozorima."""
    h, w = binary_warped.shape

    # 1) Histogram donje polovine: vrhovi su pocetne pozicije linija
    histogram = np.sum(binary_warped[h // 2:, :], axis=0)
    midpoint = w // 2
    leftx_current = int(np.argmax(histogram[:midpoint]))
    rightx_current = int(np.argmax(histogram[midpoint:]) + midpoint)

    window_height = h // nwindows
    nonzeroy, nonzerox = binary_warped.nonzero()  # koordinate svih belih piksela

    left_inds, right_inds = [], []
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # 2) Klizni prozori, od dna ka vrhu
    for window in range(nwindows):
        y_low = h - (window + 1) * window_height
        y_high = h - window * window_height
        xl_low, xl_high = leftx_current - margin, leftx_current + margin
        xr_low, xr_high = rightx_current - margin, rightx_current + margin

        cv2.rectangle(out_img, (xl_low, y_low), (xl_high, y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (xr_low, y_low), (xr_high, y_high), (0, 255, 0), 2)

        # Indeksi belih piksela koji upadaju u prozor
        good_left = ((nonzeroy >= y_low) & (nonzeroy < y_high) &
                     (nonzerox >= xl_low) & (nonzerox < xl_high)).nonzero()[0]
        good_right = ((nonzeroy >= y_low) & (nonzeroy < y_high) &
                      (nonzerox >= xr_low) & (nonzerox < xr_high)).nonzero()[0]
        left_inds.append(good_left)
        right_inds.append(good_right)

        # Ako ima dovoljno piksela, sledeci prozor centriramo na njihov prosek
        if len(good_left) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left]))
        if len(good_right) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right]))

    left_inds = np.concatenate(left_inds)
    right_inds = np.concatenate(right_inds)

    leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
    rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]

    # Bojenje pronadjenih piksela (crveno levo, plavo desno)
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(leftx, lefty, rightx, righty, height):
    """Fituje x = A*y^2 + B*y + C za obe linije. Vraca koeficijente i tacke za crtanje."""
    if len(leftx) < 3 or len(rightx) < 3:
        return None
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, height - 1, height)
    left_fitx = np.polyval(left_fit, ploty)
    right_fitx = np.polyval(right_fit, ploty)
    return left_fit, right_fit, ploty, left_fitx, right_fitx


if __name__ == "__main__":
    mtx, dist = load_calibration()
    M, Minv = get_transforms()
    out_dir = Path("output_images")
    out_dir.mkdir(exist_ok=True)
    Path("docs").mkdir(exist_ok=True)

    for p in sorted(glob.glob("test_images/*.jpg")):
        img = cv2.imread(p)
        if img.shape[:2] != IMG_SHAPE:
            continue
        name = Path(p).stem

        und = undistort(img, mtx, dist)
        binary = binary_threshold(und)["combined"]
        warped = warp(binary, M)

        leftx, lefty, rightx, righty, out_img = find_lane_pixels(warped)
        fit = fit_polynomial(leftx, lefty, rightx, righty, warped.shape[0])
        if fit is None:
            print(f"Nije pronadjena linija: {name}")
            continue
        left_fit, right_fit, ploty, left_fitx, right_fitx = fit

        plt.figure(figsize=(6, 3.4))
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color="yellow", linewidth=3)
        plt.plot(right_fitx, ploty, color="yellow", linewidth=3)
        plt.xlim(0, warped.shape[1])
        plt.ylim(warped.shape[0], 0)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_fit.png", dpi=100)
        if name == "test2":  # primer za README
            plt.savefig("docs/color_fit_lines.png", dpi=100)
        plt.close()
        print(f"Obradjeno: {name}")