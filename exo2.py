"""
stitch_panorama.py
Création d'un panorama à partir de 2 à 5 images

TP2 - Vision Artificielle
Master Informatique Visuelle
"""

import cv2
import numpy as np
import sys
import tkinter as tk
from tkinter import filedialog


# -------------------------------------------------
# Chargement des images
# -------------------------------------------------
def load_images(paths):

    images = []

    for p in paths:

        img = cv2.imread(p)

        if img is None:
            print(f"[!] Impossible de charger: {p}")
            continue

        print(f"[+] Image chargée: {p} -> {img.shape}")

        images.append(img)

    return images


# -------------------------------------------------
# Stitching manuel (SIFT + FLANN + RANSAC)
# -------------------------------------------------
def stitch_manual(images):

    if len(images) < 2:
        print("[!] Il faut au moins 2 images.")
        return None

    sift = cv2.SIFT_create()

    FLANN_INDEX_KDTREE = 1

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    result = images[0]

    for i in range(1, len(images)):

        img_left = result
        img_right = images[i]

        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        kp1, des1 = sift.detectAndCompute(gray_left, None)
        kp2, des2 = sift.detectAndCompute(gray_right, None)

        print(f"Image {i}: {len(kp1)} kp gauche | {len(kp2)} kp droite")

        if des1 is None or des2 is None:
            return result

        matches = flann.knnMatch(des2, des1, k=2)

        good = []

        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        print("Bons matches:", len(good))

        if len(good) < 10:
            print("[!] Pas assez de correspondances.")
            return result

        src_pts = np.float32(
            [kp2[m.queryIdx].pt for m in good]
        ).reshape(-1, 1, 2)

        dst_pts = np.float32(
            [kp1[m.trainIdx].pt for m in good]
        ).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            return result

        h_left, w_left = img_left.shape[:2]
        h_right, w_right = img_right.shape[:2]

        corners_right = np.float32([
            [0,0],
            [0,h_right],
            [w_right,h_right],
            [w_right,0]
        ]).reshape(-1,1,2)

        corners_trans = cv2.perspectiveTransform(corners_right, H)

        corners_left = np.float32([
            [0,0],
            [0,h_left],
            [w_left,h_left],
            [w_left,0]
        ]).reshape(-1,1,2)

        all_corners = np.concatenate((corners_left, corners_trans), axis=0)

        xmin, ymin = np.int32(all_corners.min(axis=0).ravel())
        xmax, ymax = np.int32(all_corners.max(axis=0).ravel())

        tx = -xmin
        ty = -ymin

        T = np.array([
            [1,0,tx],
            [0,1,ty],
            [0,0,1]
        ])

        pano_w = xmax - xmin
        pano_h = ymax - ymin

        warped_right = cv2.warpPerspective(img_right, T @ H, (pano_w, pano_h))
        warped_left = cv2.warpPerspective(img_left, T, (pano_w, pano_h))

        mask_left = (warped_left.sum(axis=2) > 0)
        mask_right = (warped_right.sum(axis=2) > 0)

        result = warped_right.copy()
        result[mask_left] = warped_left[mask_left]

        overlap = mask_left & mask_right

        if overlap.sum() > 0:

            cols = np.where(overlap.any(axis=0))[0]

            if len(cols) > 0:

                start = cols[0]
                end = cols[-1]

                width = end - start + 1

                alpha = np.linspace(1,0,width)

                for idx,c in enumerate(range(start,end+1)):

                    rows = np.where(overlap[:,c])[0]

                    if len(rows)>0:

                        a = alpha[idx]

                        result[rows,c] = (
                            a*warped_left[rows,c] +
                            (1-a)*warped_right[rows,c]
                        ).astype(np.uint8)

        print("Panorama partiel:", result.shape)

    return result


# -------------------------------------------------
# Stitch automatique OpenCV
# -------------------------------------------------
def stitch_opencv(images):

    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)

    status, pano = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:

        print("[+] Stitch réussi avec OpenCV")

        return pano

    else:

        print("[!] Échec Stitcher OpenCV")

        return None


# -------------------------------------------------
# Recadrage des bords noirs
# -------------------------------------------------
def crop_black_borders(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)

    contours,_ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return img

    x,y,w,h = cv2.boundingRect(max(contours,key=cv2.contourArea))

    return img[y:y+h, x:x+w]


# -------------------------------------------------
# Programme principal
# -------------------------------------------------
def main():

    root = tk.Tk()
    root.withdraw()

    chemins = filedialog.askopenfilenames(

        title="Selectionnez 2 a 5 images",

        filetypes=[("Images","*.jpg *.jpeg *.png *.bmp *.tiff")]
    )

    if not chemins:
        print("[!] Aucune image sélectionnée")
        sys.exit(1)

    paths = list(chemins)

    print("\n==============================")
    print("Panorama avec",len(paths),"images")
    print("==============================\n")

    images = load_images(paths)

    if len(images)<2:
        print("[!] Pas assez d'images")
        sys.exit(1)

    print("\nChoisissez la méthode")
    print("1 - Automatique (OpenCV)")
    print("2 - Manuelle (SIFT)")

    choix = input("Votre choix (1/2): ")

    if choix == "2":

        result = stitch_manual(images)

    else:

        result = stitch_opencv(images)

        if result is None:

            print("Tentative méthode manuelle...")

            result = stitch_manual(images)

    if result is None:

        print("[!] Panorama échoué")
        sys.exit(1)

    result = crop_black_borders(result)

    cv2.imwrite("panorama.jpg",result)

    print("\nPanorama sauvegardé: panorama.jpg")

    display = result.copy()

    if display.shape[1] > 1200:

        scale = 1200/display.shape[1]

        display = cv2.resize(
            display,
            (1200,int(display.shape[0]*scale))
        )

    cv2.imshow("Panorama",display)

    print("\nAppuyez sur une touche pour fermer")

    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()