from pathlib import Path
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Broj unutrasnjih uglova sahovske table (kolone, redovi)
CHESSBOARD = (9, 6)


def calibrate_camera(image_dir="camera_cal"):
    """Racuna matricu kamere i koeficijente distorzije iz slika sahovske table."""
    # 3D tacke table u "svetu": (0,0,0), (1,0,0), ... z je uvek 0 jer je tabla ravna
    objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)

    objpoints, imgpoints = [], []
    img_size = None
    images = sorted(glob.glob(str(Path(image_dir) / "calibration*.jpg")))

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]  # (sirina, visina)

        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD, None)
        if found:
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print(f"Uglovi nisu pronadjeni: {fname}")

    print(f"Uspesno: {len(objpoints)} od {len(images)} slika")

    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )
    print(f"RMS greska reprojekcije: {rms:.3f} px")
    return mtx, dist


def undistort(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)


if __name__ == "__main__":
    mtx, dist = calibrate_camera()
    np.savez("calibration.npz", mtx=mtx, dist=dist)

    # Primer za README: original i ispravljena slika jedna pored druge
    Path("docs").mkdir(exist_ok=True)
    img = cv2.imread("camera_cal/calibration1.jpg")
    fixed = undistort(img, mtx, dist)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original")
    ax[1].imshow(cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Undistorted")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig("docs/undistort_output.png", dpi=100)
    print("Sacuvano: docs/undistort_output.png")