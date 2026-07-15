"""
stitch_panorama.py - Création d'une image panoramique à partir de plusieurs images
Auteur: TP2 - Vision Artificielle - Master Informatique Visuelle - USTHB

Usage:
    python stitch_panorama.py image1.jpg image2.jpg image3.jpg ...
    python stitch_panorama.py --folder ./images/    (toutes les images d'un dossier)
"""

import cv2
import numpy as np
import sys
import os
import glob
import argparse


def load_images(paths):
    """Charge les images depuis une liste de chemins."""
    images = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"[!] Impossible de charger: {p}")
            continue
        print(f"[+] Image chargée: {p} -> {img.shape}")
        images.append(img)
    return images


def stitch_manual(images):
    """
    Méthode manuelle : détection SIFT + matching FLANN + RANSAC + warpPerspective.
    Coud les images deux par deux de gauche à droite.
    """
    if len(images) < 2:
        print("[!] Il faut au moins 2 images.")
        return None

    # Détecteur de points clés
    sift = cv2.SIFT_create()

    # Paramètres FLANN
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

        # Détection des keypoints et descripteurs
        kp1, des1 = sift.detectAndCompute(gray_left, None)
        kp2, des2 = sift.detectAndCompute(gray_right, None)

        print(f"  Image {i}: {len(kp1)} kp (gauche) | {len(kp2)} kp (droite)")

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            print("[!] Pas assez de points clés détectés.")
            return result

        # Matching avec FLANN + test de Lowe
        matches = flann.knnMatch(des2, des1, k=2)
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        print(f"  Bons matches: {len(good)}")

        if len(good) < 10:
            print("[!] Pas assez de bons matches. Essayez des images avec plus de chevauchement.")
            return result

        # Extraction des points correspondants
        src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Calcul de la homographie avec RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            print("[!] Homographie non trouvée.")
            return result

        inliers = np.sum(mask)
        print(f"  Inliers RANSAC: {inliers}/{len(good)}")

        # Dimensions du résultat
        h_left, w_left = img_left.shape[:2]
        h_right, w_right = img_right.shape[:2]

        # Coins de l'image droite transformés
        corners_right = np.float32([
            [0, 0], [0, h_right], [w_right, h_right], [w_right, 0]
        ]).reshape(-1, 1, 2)
        corners_transformed = cv2.perspectiveTransform(corners_right, H)

        # Calcul de la taille du panorama
        all_corners = np.concatenate([
            np.float32([[0, 0], [0, h_left], [w_left, h_left], [w_left, 0]]).reshape(-1, 1, 2),
            corners_transformed
        ], axis=0)

        x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
        x_max, y_max = np.int32(all_corners.max(axis=0).ravel())

        # Translation pour éviter les coordonnées négatives
        tx = -x_min if x_min < 0 else 0
        ty = -y_min if y_min < 0 else 0
        T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)

        # Dimensions du panorama
        pano_w = x_max - x_min
        pano_h = y_max - y_min

        # Limiter la taille maximale
        max_dim = 8000
        if pano_w > max_dim or pano_h > max_dim:
            scale = max_dim / max(pano_w, pano_h)
            pano_w = int(pano_w * scale)
            pano_h = int(pano_h * scale)
            T[:2, :] *= scale
            H = T @ H
        else:
            H = T @ H

        # Warp de l'image droite
        warped_right = cv2.warpPerspective(img_right, H, (pano_w, pano_h))

        # Copie de l'image gauche avec translation
        warped_left = cv2.warpPerspective(img_left, T, (pano_w, pano_h))

        # Fusion simple : priorité à l'image gauche là où elle existe
        mask_left = (warped_left.sum(axis=2) > 0).astype(np.uint8)
        mask_right = (warped_right.sum(axis=2) > 0).astype(np.uint8)

        # Zone de chevauchement : blending linéaire
        overlap = (mask_left & mask_right).astype(np.float32)

        result = warped_right.copy()
        result[mask_left == 1] = warped_left[mask_left == 1]

        # Blending dans la zone de chevauchement
        if overlap.sum() > 0:
            cols_overlap = np.where(overlap.any(axis=0))[0]
            if len(cols_overlap) > 0:
                col_start = cols_overlap[0]
                col_end = cols_overlap[-1]
                width_blend = col_end - col_start + 1
                if width_blend > 1:
                    alpha = np.linspace(1, 0, width_blend)
                    for c_idx, c in enumerate(range(col_start, col_end + 1)):
                        rows = np.where(overlap[:, c] > 0)[0]
                        if len(rows) > 0:
                            a = alpha[c_idx]
                            result[rows, c] = (
                                a * warped_left[rows, c].astype(np.float32) +
                                (1 - a) * warped_right[rows, c].astype(np.float32)
                            ).astype(np.uint8)

        print(f"  Panorama partiel: {result.shape}")

    return result


def stitch_opencv(images):
    """
    Méthode automatique avec cv2.Stitcher (plus simple, très robuste).
    """
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    status, pano = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        print("[+] Stitching réussi avec cv2.Stitcher!")
        return pano
    else:
        codes = {
            cv2.Stitcher_ERR_NEED_MORE_IMGS: "Pas assez d'images",
            cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Échec estimation homographie",
            cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Échec ajustement caméra",
        }
        msg = codes.get(status, f"Code erreur: {status}")
        print(f"[!] cv2.Stitcher échoué: {msg}")
        return None


def crop_black_borders(img):
    """Recadre les bords noirs du panorama."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return img[y:y+h, x:x+w]


def main():
    parser = argparse.ArgumentParser(description="Panorama multi-images OpenCV")
    parser.add_argument("images", nargs="*", help="Chemins des images")
    parser.add_argument("--folder", "-f", help="Dossier contenant les images")
    parser.add_argument("--output", "-o", default="panorama.jpg", help="Fichier de sortie")
    parser.add_argument("--method", "-m", choices=["auto", "manual"], default="auto",
                        help="Méthode: 'auto' (cv2.Stitcher) ou 'manual' (SIFT+RANSAC)")
    parser.add_argument("--no-crop", action="store_true", help="Ne pas recadrer les bords noirs")
    args = parser.parse_args()

    # Collecte des chemins d'images
    paths = []
    if args.folder:
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            paths += sorted(glob.glob(os.path.join(args.folder, ext)))
    if args.images:
        paths += args.images

    if not paths:
        print("Usage: python stitch_panorama.py img1.jpg img2.jpg img3.jpg ...")
        print("   ou: python stitch_panorama.py --folder ./mes_images/")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"  Panorama de {len(paths)} images | méthode: {args.method}")
    print(f"{'='*50}\n")

    images = load_images(paths)
    if len(images) < 2:
        print("[!] Il faut au moins 2 images valides.")
        sys.exit(1)

    # Stitching
    if args.method == "auto":
        result = stitch_opencv(images)
        if result is None:
            print("[*] Tentative avec la méthode manuelle...")
            result = stitch_manual(images)
    else:
        result = stitch_manual(images)

    if result is None:
        print("[!] Échec du panorama.")
        sys.exit(1)

    # Recadrage
    if not args.no_crop:
        result = crop_black_borders(result)
        print(f"[+] Après recadrage: {result.shape}")

    # Sauvegarde
    out_path = args.output
    cv2.imwrite(out_path, result)
    print(f"\n[✓] Panorama sauvegardé: {out_path}")
    print(f"    Taille finale: {result.shape[1]}x{result.shape[0]} px")

    # Affichage
    display = result.copy()
    max_display = 1200
    if display.shape[1] > max_display:
        scale = max_display / display.shape[1]
        display = cv2.resize(display, (max_display, int(display.shape[0] * scale)))

    cv2.imshow("Panorama", display)
    print("\nAppuyez sur une touche pour fermer...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()