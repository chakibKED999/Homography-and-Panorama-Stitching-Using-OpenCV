import cv2
import numpy as np
import os

# ============================================================
# FONCTION D'AFFICHAGE : redimensionne en gardant l'aspect ratio
# ============================================================
MAX_DISPLAY_W = 1280
MAX_DISPLAY_H = 720

def imshow(title, img):
    h, w = img.shape[:2]
    scale = min(MAX_DISPLAY_W / w, MAX_DISPLAY_H / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cv2.imshow(title, img)

# ============================================================
# LISTE DES IMAGES (peu importe l'ordre, le code les trie)
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))

image_files = ['image1.jpg', 'image2.jpg', 'image3.jpg' ]

# ============================================================
# ETAPE 1 : Charger toutes les images
# cv2.imread ne supporte pas les chemins arabes/unicode sur Windows
# Solution : lire en bytes avec numpy puis decoder avec cv2.imdecode
# ============================================================
def imread_unicode(path):
    stream = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(stream, cv2.IMREAD_COLOR)

images = []
for f in image_files:
    full_path = os.path.join(script_dir, f)
    img = imread_unicode(full_path)
    if img is not None:
        images.append(img)
        print(f"Chargee : {f}")
    else:
        print(f"ERREUR : impossible de lire {full_path}")

print(f"\nNombre d'images chargees : {len(images)}")

# ============================================================
# ETAPE 2 : Detecter l'ordre GAUCHE -> DROITE automatiquement
#
# LOGIQUE :
#   Pour chaque paire (i, j), on calcule l'homographie H de i vers j.
#   H[0,2] = translation horizontale :
#     - H[0,2] > 0  => image i est a DROITE  => son score monte
#     - H[0,2] < 0  => image i est a GAUCHE  => son score descend
#   np.argsort(scores) => ordre croissant = gauche vers droite
# ============================================================
n = len(images)
scores = np.zeros(n)

sift = cv2.SIFT_create()

kps  = []
dess = []
for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    kps.append(kp)
    dess.append(des)

matcher_order = cv2.BFMatcher()
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        matches_ij = matcher_order.knnMatch(dess[i], dess[j], k=2)
        good_ij = [m for m, nn in matches_ij if m.distance < 0.3 * nn.distance]
        if len(good_ij) < 10:
            continue
        src_pts = np.float32([kps[i][m.queryIdx].pt for m in good_ij]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps[j][m.trainIdx].pt for m in good_ij]).reshape(-1, 1, 2)
        M_ij, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M_ij is None:
            continue
        scores[i] += M_ij[0, 2]

order = np.argsort(scores)
images_orig = [images[k]     for k in order]
names_orig  = [image_files[k] for k in order]

print("\nOrdre detecte (gauche -> droite) :")
for rank, (name, k) in enumerate(zip(names_orig, order)):
    print(f"  {rank+1}. {name}  (score = {scores[k]:.1f})")

# ============================================================
# ETAPE 3 : Assemblage iteratif
#
# LOGIQUE :
#   - panorama = image la plus a gauche
#   - remaining = les autres images (dans l'ordre gauche->droite)
#   - A chaque etape :
#       1. Le PANORAMA cherche dans remaining quelle image match le mieux
#       2. On stitch panorama + best_match  => nouveau panorama temporaire
#       3. On retire best_match de remaining
#       4. On repete jusqu'a ce que remaining soit vide
#   => Le panorama final est sauvegarde
#
# POURQUOI 0.75 ICI (et pas 0.3) ?
#   0.3 est tres strict, parfait pour le tri initial sur images originales.
#   Mais le panorama contient des zones deformees/floues apres le warp,
#   donc SIFT trouve moins de features nettes => on utilise 0.75
#   (seuil standard de Lowe) pour rester robuste.
# ============================================================

def trim(frame):
    # Version iterative (evite RecursionError sur les grands panoramas)
    # crop top
    while frame.shape[0] > 0 and not np.sum(frame[0]):
        frame = frame[1:]
    # crop bottom
    while frame.shape[0] > 0 and not np.sum(frame[-1]):
        frame = frame[:-1]
    # crop left
    while frame.shape[1] > 0 and not np.sum(frame[:, 0]):
        frame = frame[:, 1:]
    # crop right
    while frame.shape[1] > 0 and not np.sum(frame[:, -1]):
        frame = frame[:, :-1]
    return frame

MIN_MATCH_COUNT = 10

# Panorama initial = image la plus a gauche
panorama   = images_orig[0]
remaining  = list(zip(names_orig[1:], images_orig[1:]))  # (nom, image)
step       = 1

while len(remaining) > 0:

    print(f"\n=== Etape {step} : panorama cherche la meilleure image dans remaining ===")
    print(f"    Images restantes : {[r[0] for r in remaining]}")

    # --- Calculer features du panorama courant ---
    imgr_pano = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    kp_pano, des_pano = sift.detectAndCompute(imgr_pano, None)

    # --- Trouver quelle image de remaining match le mieux le panorama ---
    best_idx     = 0
    best_good    = []
    best_kp_r    = None
    best_des_r   = None

    for idx, (name_r, img_r) in enumerate(remaining):
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        kp_r, des_r = sift.detectAndCompute(gray_r, None)

        match2   = cv2.BFMatcher()
        matches2 = match2.knnMatch(des_r, des_pano, k=2)
        good     = [m for m, nn in matches2 if m.distance < 0.75 * nn.distance]

        print(f"    {name_r} : {len(good)} matches avec le panorama")

        if len(good) > len(best_good):
            best_good  = good
            best_idx   = idx
            best_kp_r  = kp_r
            best_des_r = des_r

    best_name, img_r = remaining[best_idx]
    print(f"\n    => Meilleure image : {best_name} ({len(best_good)} matches)")

    # --- Exactement le code original de stitch.py ---
    imgr = cv2.cvtColor(img_r,    cv2.COLOR_BGR2GRAY)
    imgl = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)

    kpr, desr = sift.detectAndCompute(imgr, None)
    kpl, desl = sift.detectAndCompute(imgl, None)

    imshow('original_image_left_keypoints',  cv2.drawKeypoints(panorama, kpl, None))
    cv2.waitKey(0)

    imshow('original_image_right_keypoints', cv2.drawKeypoints(img_r, kpr, None))
    cv2.waitKey(0)

    match3   = cv2.BFMatcher()
    matches3 = match3.knnMatch(desr, desl, k=2)

    good = []
    for m, nn in matches3:
        if m.distance < 0.75 * nn.distance:
            good.append(m)

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       flags=2)

    img3 = cv2.drawMatches(img_r, kpr, panorama, kpl, good, None, **draw_params)
    imshow(f"Draw Matches Left Right - etape {step}", img3)
    cv2.waitKey(0)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kpr[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpl[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w = imgr.shape
        pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1, 1, 2)
        dst_overlap = cv2.perspectiveTransform(pts, M)
        imgr_poly   = cv2.polylines(imgr.copy(), [np.int32(dst_overlap)], True, 255, 3, cv2.LINE_AA)
        imshow(f"right_image_overlapping - etape {step}", imgr_poly)
        cv2.waitKey(0)

        print(panorama.shape[1] + img_r.shape[1], panorama.shape[0])
        dst = cv2.warpPerspective(img_r, M, (panorama.shape[1] + img_r.shape[1], panorama.shape[0]))
        imshow(f"dst warpPerspective - etape {step}", dst)
        cv2.waitKey(0)

        dst[0:panorama.shape[0], 0:panorama.shape[1]] = panorama
        imshow(f"add left image to the warped right - etape {step}", dst)
        cv2.waitKey(0)

        # Le panorama temporaire devient le nouveau panorama
        panorama = dst
        print(f"    => Panorama temporaire cree (etape {step})")

    else:
        print(f"Not enough matches found - {len(good)}/{MIN_MATCH_COUNT}")

    # Retirer l'image utilisee de remaining
    remaining.pop(best_idx)
    step += 1

# ============================================================
# ETAPE 4 : Rogner les bords noirs et sauvegarder le panorama final
# ============================================================

arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(arr[:, -1])
print(arr[:, :-2])
print(arr[:,0])
print(arr[:,1:])
print(arr[-1])
print(arr[0:-1])
print(arr[0])
print(arr[1:])

imshow("original_image_stitched_crop.jpg", trim(panorama))
cv2.waitKey(0)
cv2.imwrite("original_image_stitched_crop.jpg", trim(panorama))
print("\nPanorama final sauvegarde : original_image_stitched_crop.jpg")