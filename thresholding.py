from pathlib import Path
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt

from calibration import undistort

# Rezolucija slika za koje je kamera kalibrisana (visina, sirina)
IMG_SHAPE = (720, 1280)


def load_calibration(path="calibration.npz"):
    """Ucitava sacuvanu matricu kamere i koeficijente distorzije."""
    data = np.load(path)
    return data["mtx"], data["dist"]


def binary_threshold(img_bgr, s_thresh=(170, 255), sx_thresh=(20, 100),
                     l_thresh=(200, 255), use_l=True):
    """Pravi binarnu sliku (0/1) iz S kanala, Sobel gradijenta po x i (opciono) L kanala.

    Vraca recnik sa medjurezultatima i krajnjom slikom pod kljucem "combined".
    """
    hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)  # OpenCV ucitava slike kao BGR!

    # 1) Boja: S kanal (zasicenost) - dobro hvata zute linije
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # 2) Boja: L kanal (svetlost) - hvata bele linije
    l_channel = hls[:, :, 1]
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1

    # 3) Gradijent: Sobel po x na sivoj slici
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    # Skaliranje na 0-255, da pragovi ne zavise od osvetljenja slike
    scaled = np.uint8(255 * abs_sobelx / max(np.max(abs_sobelx), 1e-6))
    sx_binary = np.zeros_like(scaled)
    sx_binary[(scaled >= sx_thresh[0]) & (scaled <= sx_thresh[1])] = 1

    # 4) Spajanje: piksel ostaje ako ga je uhvatio bilo koji metod
    combined = np.zeros_like(sx_binary)
    mask = (s_binary == 1) | (sx_binary == 1)
    if use_l:
        mask = mask | (l_binary == 1)
    combined[mask] = 1

    return {"combined": combined, "s": s_binary, "l": l_binary, "sx": sx_binary}


if __name__ == "__main__":
    mtx, dist = load_calibration()
    out_dir = Path("output_images")
    out_dir.mkdir(exist_ok=True)
    Path("docs").mkdir(exist_ok=True)

    paths = sorted(glob.glob("test_images/*.jpg"))
    if not paths:
        raise SystemExit("Nema slika u folderu test_images")

    # Obrada svih test slika (samo onih sa rezolucijom za koju je kamera kalibrisana)
    for p in paths:
        img = cv2.imread(p)
        if img.shape[:2] != IMG_SHAPE:
            print(f"Preskoceno (druga rezolucija {img.shape[1]}x{img.shape[0]}): {Path(p).name}")
            continue
        und = undistort(img, mtx, dist)
        result = binary_threshold(und)
        name = Path(p).stem
        cv2.imwrite(str(out_dir / f"{name}_undistorted.jpg"), und)
        cv2.imwrite(str(out_dir / f"{name}_binary.png"), result["combined"] * 255)
        print(f"Obradjeno: {name}")

    # Slike za README, na jednoj test slici
    example = Path("test_images/test1.jpg")
    if not example.exists():
        example = Path(paths[0])
    img = cv2.imread(str(example))
    und = undistort(img, mtx, dist)
    with_l = binary_threshold(und, use_l=True)
    without_l = binary_threshold(und, use_l=False)

    # Slika 1: original i ispravljena slika puta
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original")
    ax[1].imshow(cv2.cvtColor(und, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Undistorted")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig("docs/undistorted_road.png", dpi=100)
    plt.close()

    # Slika 2: koraci binarizacije, sa i bez L kanala
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    ax[0, 0].imshow(cv2.cvtColor(und, cv2.COLOR_BGR2RGB))
    ax[0, 0].set_title("Undistorted")
    ax[0, 1].imshow(with_l["s"], cmap="gray")
    ax[0, 1].set_title("S kanal (boja)")
    ax[0, 2].imshow(with_l["l"], cmap="gray")
    ax[0, 2].set_title("L kanal (svetlost)")
    ax[1, 0].imshow(with_l["sx"], cmap="gray")
    ax[1, 0].set_title("Sobel x (gradijent)")
    ax[1, 1].imshow(without_l["combined"], cmap="gray")
    ax[1, 1].set_title("Kombinovano bez L")
    ax[1, 2].imshow(with_l["combined"], cmap="gray")
    ax[1, 2].set_title("Kombinovano sa L")
    for a in ax.ravel():
        a.axis("off")
    plt.tight_layout()
    plt.savefig("docs/binary_example.png", dpi=100)
    plt.close()

    print(f"Slike za README sacuvane u docs/ (primer: {example.name})")