from pathlib import Path
import glob

import cv2
import numpy as np

from calibration import undistort
from thresholding import load_calibration, binary_threshold, IMG_SHAPE
from perspective import get_transforms, warp
from lane_finding import find_lane_pixels, fit_polynomial
from measurements import measure_lane

# Ucitava se jednom, pa se koristi za svaku sliku (vazno za video)
MTX, DIST = load_calibration()
M, MINV = get_transforms()


def draw_lane(und, warped_shape, ploty, left_fitx, right_fitx, minv):
    """Popunjava prostor izmedju linija zelenom bojom i vraca ga na originalnu sliku."""
    color_warp = np.zeros((warped_shape[0], warped_shape[1], 3), dtype=np.uint8)

    # Leva linija ide od vrha ka dnu, desna od dna ka vrhu, da poligon bude zatvoren
    pts_left = np.transpose(np.vstack([left_fitx, ploty]))
    pts_right = np.flipud(np.transpose(np.vstack([right_fitx, ploty])))
    pts = np.vstack([pts_left, pts_right])
    cv2.fillPoly(color_warp, [np.int32(pts)], (0, 255, 0))

    # Inverzna perspektiva: iz pogleda odozgo nazad u pogled kamere
    h, w = und.shape[:2]
    newwarp = cv2.warpPerspective(color_warp, minv, (w, h))
    return cv2.addWeighted(und, 1.0, newwarp, 0.3, 0)


def process_image(img):
    """Ceo pipeline na jednoj slici (BGR, 1280x720). Vraca sliku sa ucrtanom trakom."""
    und = undistort(img, MTX, DIST)
    binary = binary_threshold(und)["combined"]
    warped = warp(binary, M)

    leftx, lefty, rightx, righty, _ = find_lane_pixels(warped)
    fit = fit_polynomial(leftx, lefty, rightx, righty, warped.shape[0])
    if fit is None:
        return und  # nije nadjena linija, vracamo samo ispravljenu sliku
    left_fit, right_fit, ploty, left_fitx, right_fitx = fit

    result = draw_lane(und, warped.shape, ploty, left_fitx, right_fitx, MINV)

    h, w = warped.shape
    curv, off, _ = measure_lane(leftx, lefty, rightx, righty, w, h)
    curv_text = "Curvature radius: straight" if curv > 5000 else f"Curvature radius: {curv:.0f} m"
    side = "right" if off > 0 else "left"
    cv2.putText(result, curv_text, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(result, f"Vehicle is {abs(off):.2f} m {side} of center", (40, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    return result


if __name__ == "__main__":
    out_dir = Path("output_images")
    out_dir.mkdir(exist_ok=True)
    Path("docs").mkdir(exist_ok=True)

    for p in sorted(glob.glob("test_images/*.jpg")):
        img = cv2.imread(p)
        if img.shape[:2] != IMG_SHAPE:
            continue
        name = Path(p).stem
        result = process_image(img)
        cv2.imwrite(str(out_dir / f"{name}_result.jpg"), result)
        if name == "test2":  # primer za README
            cv2.imwrite("docs/example_output.jpg", result)
        print(f"Obradjeno: {name}")