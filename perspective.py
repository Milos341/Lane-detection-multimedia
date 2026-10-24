from pathlib import Path
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt

from calibration import undistort
from thresholding import load_calibration, binary_threshold, IMG_SHAPE

# Cetiri tacke trapeza na slici (gore-levo, gore-desno, dole-desno, dole-levo)
# POCETNE VREDNOSTI - proveriti i podesiti na straight_lines1.jpg!
SRC = np.float32([[580, 460], [705, 460], [1127, 720], [203, 720]])

# Gde te tacke treba da zavrse u pogledu odozgo (pravougaonik)
# Margine levo i desno (320 px) ostavljaju mesta da se vide krive linije
DST = np.float32([[320, 0], [960, 0], [960, 720], [320, 720]])


def get_transforms(src=SRC, dst=DST):
    """Vraca matricu transformacije M i njenu inverznu Minv.

    Za razliku od profesorovog warper-a, matrica se racuna jednom, a ne pri svakom
    pozivu, a inverzna nam treba da kasnije traku vratimo na originalnu sliku.
    """
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


def warp(img, M, flags=cv2.INTER_NEAREST):
    """Primenjuje perspektivnu transformaciju, izlaz je iste velicine kao ulaz.

    INTER_NEAREST (kao u primeru sa predavanja) ne meša susedne piksele,
    pa binarna slika ostaje cisto 0/1. Za obicne slike u boji moze INTER_LINEAR.
    """
    h, w = img.shape[:2]
    return cv2.warpPerspective(img, M, (w, h), flags=flags)


if __name__ == "__main__":
    mtx, dist = load_calibration()
    M, Minv = get_transforms()
    out_dir = Path("output_images")
    out_dir.mkdir(exist_ok=True)
    Path("docs").mkdir(exist_ok=True)

    # 1) Provera tacaka na ravnom putu
    img = cv2.imread("test_images/straight_lines1.jpg")
    if img is None:
        raise SystemExit("Nema test_images/straight_lines1.jpg")
    und = undistort(img, mtx, dist)

    marked = und.copy()
    cv2.polylines(marked, [SRC.astype(np.int32)], True, (0, 0, 255), 3)

    top_down = warp(und, M, flags=cv2.INTER_LINEAR)
    top_marked = top_down.copy()
    # Vertikalne zelene linije na DST x pozicijama: linije trake treba da ih prate
    for x in (int(DST[0][0]), int(DST[1][0])):
        cv2.line(top_marked, (x, 0), (x, IMG_SHAPE[0]), (0, 255, 0), 2)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].imshow(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Undistorted + trapez (SRC)")
    ax[1].imshow(cv2.cvtColor(top_marked, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Pogled odozgo (warped)")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig("docs/warped_straight_lines.png", dpi=100)
    plt.close()
    print("Sacuvano: docs/warped_straight_lines.png")

    # 2) Warp binarnih slika za sve test slike
    for p in sorted(glob.glob("test_images/*.jpg")):
        im = cv2.imread(p)
        if im.shape[:2] != IMG_SHAPE:
            print(f"Preskoceno (druga rezolucija {im.shape[1]}x{im.shape[0]}): {Path(p).name}")
            continue
        u = undistort(im, mtx, dist)
        binary = binary_threshold(u)["combined"]
        warped = warp(binary, M)
        cv2.imwrite(str(out_dir / f"{Path(p).stem}_warped.png"), warped * 255)
        print(f"Obradjeno: {Path(p).stem}")