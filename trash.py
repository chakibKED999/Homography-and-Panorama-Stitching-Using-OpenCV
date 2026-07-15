

#reference: https://pylessons.com/OpenCV-image-stiching-continue

import cv2
import numpy as np

# ============================================================
# LISTE DES IMAGES (peu importe l'ordre, le code les trie)
# ============================================================
image_files = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg']

# ============================================================
# ETAPE 1 : Charger toutes les images
# ============================================================
images = []
for f in image_files:
    img = cv2.imread(f)
    if img is not None:
        images.append(img)
        print(f"Chargee : {f}")
    else:
        print(f"ERREUR : impossible de lire {f}")

print(f"\nNombre d'images chargees : {len(images)}")

# ============================================================
# ETAPE 2 : Detecter l'ordre GAUCHE -> DROITE automatiquement
#
# LOGIQUE :
#   Pour chaque paire (i, j), on calcule l'homographie H de i vers j.
#   H[0,2] = translation horizontale :
#     - H[0,2] > 0  => image i est a DROITE  => son score monte
#     - H[0,2] < 0  => image i est a GAUCHE  => son score descend
#   On accumule le score pour chaque image.
#   np.argsort(scores) => ordre croissant = gauche vers droite
# ============================================================
n = len(images)
scores = np.zeros(n)

sift = cv2.SIFT_create()

# Calculer les keypoints et descripteurs pour toutes les images
kps  = []
dess = []
for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    kps.append(kp)
    dess.append(des)

# Calculer les scores de position
match = cv2.BFMatcher()
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        matches_ij = match.knnMatch(dess[i], dess[j], k=2)
        good_ij = []
        for m, nn in matches_ij:
            if m.distance < 0.3 * nn.distance:
                good_ij.append(m)

        if len(good_ij) < 10:
            continue

        src_pts = np.float32([kps[i][m.queryIdx].pt for m in good_ij]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps[j][m.trainIdx].pt for m in good_ij]).reshape(-1, 1, 2)

        M_ij, mask_ij = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M_ij is None:
            continue

        # H[0,2] = translation horizontale : positive = image i est a droite
        scores[i] += M_ij[0, 2]

# Trier les images par score (le plus petit = le plus a gauche)
order = np.argsort(scores)
images = [images[k] for k in order]
kps    = [kps[k]    for k in order]
dess   = [dess[k]   for k in order]

print("\nOrdre detecte (gauche -> droite) :")
for rank, k in enumerate(order):
    print(f"  {rank+1}. {image_files[k]}  (score = {scores[k]:.1f})")

# ============================================================
# ETAPE 3 : Assemblage sequentiel - on garde le code original
#
# On commence avec images[0] (la plus a gauche)
# et on colle images[1], images[2], ... une par une
# ============================================================

def trim(frame):
    #crop top row if empty
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom row if empty
    if not np.sum(frame[-1]):
        return trim(frame[0:-1])
    #crop left column if empty
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop right column if empty
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

# Demarrer avec l'image la plus a gauche
img_l = images[0]

for step in range(1, len(images)):

    img_r = images[step]

    # ---- Exactement le code original de stitch.py ----
    imgr = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    imgl = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)

    kpr, desr = sift.detectAndCompute(imgr, None)
    kpl, desl = sift.detectAndCompute(imgl, None)

    cv2.imshow('original_image_left_keypoints',  cv2.drawKeypoints(img_l, kpl, None))
    cv2.waitKey(0)

    cv2.imshow('original_image_right_keypoints', cv2.drawKeypoints(img_r, kpr, None))
    cv2.waitKey(0)

    match2   = cv2.BFMatcher()
    matches2 = match2.knnMatch(desr, desl, k=2)

    good = []
    for m, n in matches2:
        if m.distance < 0.3 * n.distance:
            good.append(m)

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       flags=2)

    img3 = cv2.drawMatches(img_r, kpr, img_l, kpl, good, None, **draw_params)
    cv2.imshow(f"Draw Matches - etape {step}", img3)
    cv2.waitKey(0)

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kpr[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpl[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w = imgr.shape
        pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1, 1, 2)
        dst_pts_transformed = cv2.perspectiveTransform(pts, M)
        imgr_poly = cv2.polylines(imgr.copy(), [np.int32(dst_pts_transformed)], True, 255, 3, cv2.LINE_AA)
        cv2.imshow(f"right_image_overlapping - etape {step}", imgr_poly)
        cv2.waitKey(0)
    else:
        print(f"Not enough matches found - {len(good)}/{MIN_MATCH_COUNT}")
        continue

    print(img_l.shape[1] + img_r.shape[1], img_l.shape[0])
    dst = cv2.warpPerspective(img_r, M, (img_l.shape[1] + img_r.shape[1], img_l.shape[0]))
    cv2.imshow(f"dst warpPerspective - etape {step}", dst)
    cv2.waitKey(0)

    dst[0:img_l.shape[0], 0:img_l.shape[1]] = img_l
    cv2.imshow(f"add left image to the warped right - etape {step}", dst)
    cv2.waitKey(0)

    # Le panorama courant devient l'image gauche de la prochaine etape
    img_l = dst

# ============================================================
# ETAPE 4 : Rogner les bords noirs et sauvegarder
# ============================================================

#https://www.w3schools.com/python/numpy/numpy_array_slicing.asp
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(arr[:, -1])
print(arr[:, :-2])
print(arr[:,0])
print(arr[:,1:])
print(arr[-1])
print(arr[0:-1])
print(arr[0])
print(arr[1:])

cv2.imshow("original_image_stitched_crop.jpg", trim(img_l))
cv2.waitKey(0)
cv2.imwrite("original_image_stitched_crop.jpg", trim(img_l))





















"""
#reference: https://pylessons.com/OpenCV-image-stiching-continue

import cv2
import numpy as np
import glob
import os

# ============================================================
# DOSSIER CONTENANT LES IMAGES
# ============================================================
folder = r'C:\Users\CHAKI\0\سطح المكتب\S2\2025-2026\Vision Artificielle\TP\TP2\pano'  # <-- changer par le chemin de ton dossier si besoin

# Charger toutes les jpg/png du dossier automatiquement
image_files = sorted(glob.glob(os.path.join(folder, '*.jpg')) +
                     glob.glob(os.path.join(folder, '*.png')))

images = []
for f in image_files:
    img = cv2.imread(f)
    if img is not None:
        images.append((os.path.basename(f), img))
        print(f"Chargee : {os.path.basename(f)}")
    else:
        print(f"ERREUR : impossible de lire {f}")

print(f"\nNombre d'images chargees : {len(images)}")

# ============================================================
# ETAPE 1 : Calculer les keypoints une seule fois pour toutes
# ============================================================
sift = cv2.SIFT_create()

kps  = []
dess = []
for (name, img) in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    kps.append(kp)
    dess.append(des)

# ============================================================
# ETAPE 2 : Detecter l'ordre GAUCHE -> DROITE
#
# LOGIQUE :
#   Pour chaque paire (i, j), on calcule H de i vers j.
#   H[0,2] = translation horizontale :
#     H[0,2] > 0  => image i est a DROITE => son score monte
#     H[0,2] < 0  => image i est a GAUCHE => son score descend
#   argsort(scores) => ordre croissant = gauche vers droite
# ============================================================
n = len(images)
scores = np.zeros(n)
matcher = cv2.BFMatcher()

for i in range(n):
    for j in range(n):
        if i == j:
            continue
        matches_ij = matcher.knnMatch(dess[i], dess[j], k=2)
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
images = [images[k] for k in order]
kps    = [kps[k]    for k in order]
dess   = [dess[k]   for k in order]

print("\nOrdre detecte (gauche -> droite) :")
for rank, (name, _) in enumerate(images):
    k = order[rank]
    print(f"  {rank+1}. {name}  (score = {scores[k]:.1f})")

# ============================================================
# ETAPE 3 : Calculer H entre chaque paire CONSECUTIVES d'originales
#
# POURQUOI PAS CONTRE LE PANORAMA ?
#   Apres le 1er stitch, img_l devient un panorama deforme.
#   SIFT ne trouve plus assez de features => "Not enough matches".
#   La solution : on match toujours deux images ORIGINALES voisines.
#
# H[i] mappe images[i+1] -> repere de images[i]
# ============================================================
homographies = []

for step in range(len(images) - 1):
    name_l, img_l = images[step]
    name_r, img_r = images[step + 1]

    print(f"\n--- Paire {step+1} : {name_l}  <->  {name_r} ---")

    imgr = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    imgl = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)

    kpr, desr = sift.detectAndCompute(imgr, None)
    kpl, desl = sift.detectAndCompute(imgl, None)

    cv2.imshow('original_image_left_keypoints',  cv2.drawKeypoints(img_l, kpl, None))
    cv2.waitKey(0)
    cv2.imshow('original_image_right_keypoints', cv2.drawKeypoints(img_r, kpr, None))
    cv2.waitKey(0)

    match2   = cv2.BFMatcher()
    matches2 = match2.knnMatch(desr, desl, k=2)

    good = []
    for m, nn in matches2:
        if m.distance < 0.3 * nn.distance:
            good.append(m)

    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, flags=2)
    img3 = cv2.drawMatches(img_r, kpr, img_l, kpl, good, None, **draw_params)
    cv2.imshow(f"Draw Matches - paire {step+1}", img3)
    cv2.waitKey(0)

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kpr[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpl[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w = imgr.shape
        pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1, 1, 2)
        dst_overlap = cv2.perspectiveTransform(pts, M)
        imgr_poly = cv2.polylines(imgr.copy(), [np.int32(dst_overlap)], True, 255, 3, cv2.LINE_AA)
        cv2.imshow(f"right_image_overlapping - paire {step+1}", imgr_poly)
        cv2.waitKey(0)

        homographies.append(M)
        print(f"  Matches : {len(good)}  => H calculee")
    else:
        print(f"Not enough matches found - {len(good)}/{MIN_MATCH_COUNT}")
        homographies.append(np.eye(3))  # pas de match => pas de deplacement

# ============================================================
# ETAPE 4 : Composer les homographies (cumulatif)
#
# H_cum[0] = I              => images[0] reste dans son repere
# H_cum[1] = H[0]           => images[1] -> repere de images[0]
# H_cum[2] = H[0] @ H[1]   => images[2] -> images[1] -> images[0]
# H_cum[k] = H[0] @ ... @ H[k-1]
#
# Ainsi chaque image est projetee dans le repere commun (images[0])
# sans jamais matcher contre un panorama deforme.
# ============================================================
H_cum = [np.eye(3)]
for H in homographies:
    H_cum.append(H_cum[-1] @ H)

# Trouver la taille du canvas en projetant les coins de chaque image
all_corners = []
for i, (_, img) in enumerate(images):
    h, w = img.shape[:2]
    corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(corners, H_cum[i])
    all_corners.append(transformed)

all_corners_np = np.concatenate(all_corners, axis=0)
x_min = int(np.floor(all_corners_np[:, :, 0].min()))
y_min = int(np.floor(all_corners_np[:, :, 1].min()))
x_max = int(np.ceil(all_corners_np[:, :, 0].max()))
y_max = int(np.ceil(all_corners_np[:, :, 1].max()))

# Translation pour coords positives
T = np.array([[1, 0, -x_min],
              [0, 1, -y_min],
              [0, 0,       1]], dtype=np.float64)

canvas_w = x_max - x_min
canvas_h = y_max - y_min
print(f"\nCanvas final : {canvas_w} x {canvas_h}")

# ============================================================
# ETAPE 5 : Warper chaque image sur le canvas
#
# Meme principe que le code original :
#   dst = warpPerspective(img_r, M, ...)
#   dst[0:h, 0:w] = img_l
# Ici on fait la meme chose pour toutes les images,
# en collant de droite vers gauche (l'image gauche reste au-dessus).
# ============================================================
dst = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

for i in range(len(images) - 1, -1, -1):   # droite -> gauche
    _, img = images[i]
    H_final = T @ H_cum[i]
    warped = cv2.warpPerspective(img, H_final, (canvas_w, canvas_h))
    mask = np.any(warped > 0, axis=2)
    dst[mask] = warped[mask]

cv2.imshow("add left image to the warped right.jpg", dst)
cv2.waitKey(0)

# ============================================================
# ETAPE 6 : Rogner les bords noirs (trim original)
# ============================================================
def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom
    if not np.sum(frame[-1]):
        return trim(frame[0:-1])
    #crop left
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop right
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

#https://www.w3schools.com/python/numpy/numpy_array_slicing.asp
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(arr[:, -1])
print(arr[:, :-2])
print(arr[:,0])
print(arr[:,1:])
print(arr[-1])
print(arr[0:-1])
print(arr[0])
print(arr[1:])

cv2.imshow("original_image_stitched_crop.jpg", trim(dst))
cv2.waitKey(0)
cv2.imwrite("original_image_stitched_crop.jpg", trim(dst))
print("\nPanorama sauvegarde : original_image_stitched_crop.jpg")
"""










"""

# -*- coding: utf-8 -*-
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
    scale = min(MAX_DISPLAY_W / w, MAX_DISPLAY_H / h, 1.0)  # jamais agrandir
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cv2.imshow(title, img)

# ============================================================
# LISTE DES IMAGES (peu importe l'ordre, le code les trie)
# ============================================================
# os.path.abspath(__file__) resout le probleme du chemin sur Windows
script_dir = os.path.dirname(os.path.abspath(__file__))

image_files = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg']

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
#   On accumule le score pour chaque image.
#   np.argsort(scores) => ordre croissant = gauche vers droite
# ============================================================
n = len(images)
scores = np.zeros(n)

sift = cv2.SIFT_create()

# Calculer les keypoints et descripteurs pour toutes les images
kps  = []
dess = []
for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    kps.append(kp)
    dess.append(des)

# Calculer les scores de position
match = cv2.BFMatcher()
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        matches_ij = match.knnMatch(dess[i], dess[j], k=2)
        good_ij = []
        for m, nn in matches_ij:
            if m.distance < 0.3 * nn.distance:
                good_ij.append(m)

        if len(good_ij) < 10:
            continue

        src_pts = np.float32([kps[i][m.queryIdx].pt for m in good_ij]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps[j][m.trainIdx].pt for m in good_ij]).reshape(-1, 1, 2)

        M_ij, mask_ij = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M_ij is None:
            continue

        # H[0,2] = translation horizontale : positive = image i est a droite
        scores[i] += M_ij[0, 2]

# Trier les images par score (le plus petit = le plus a gauche)
order = np.argsort(scores)
images_orig = [images[k] for k in order]   # images originales triees (jamais modifiees)
kps          = [kps[k]   for k in order]
dess         = [dess[k]  for k in order]

print("\nOrdre detecte (gauche -> droite) :")
for rank, k in enumerate(order):
    print(f"  {rank+1}. {image_files[k]}  (score = {scores[k]:.1f})")

# ============================================================
# ETAPE 3 : Calculer H entre chaque paire CONSECUTIVES d'ORIGINALES
#
# POURQUOI PAS CONTRE LE PANORAMA ?
#   Si on fait knnMatch(desr, desl) ou desl vient du panorama deja warpe,
#   SIFT trouve peu de features dans les zones deformees/noires
#   => "Not enough matches" a partir de la 3eme image.
#
#   Solution : on calcule H toujours entre deux images ORIGINALES voisines.
#   H[i] mappe images_orig[i+1] -> repere de images_orig[i]
# ============================================================
homographies = []

for step in range(len(images_orig) - 1):

    img_l = images_orig[step]
    img_r = images_orig[step + 1]

    # ---- Exactement le code original de stitch.py ----
    imgr = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    imgl = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)

    kpr, desr = sift.detectAndCompute(imgr, None)
    kpl, desl = sift.detectAndCompute(imgl, None)

    imshow('original_image_left_keypoints',  cv2.drawKeypoints(img_l, kpl, None))
    cv2.waitKey(0)

    imshow('original_image_right_keypoints', cv2.drawKeypoints(img_r, kpr, None))
    cv2.waitKey(0)

    match2   = cv2.BFMatcher()
    matches2 = match2.knnMatch(desr, desl, k=2)

    good = []
    for m, nn in matches2:
        if m.distance < 0.3 * nn.distance:
            good.append(m)

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       flags=2)

    img3 = cv2.drawMatches(img_r, kpr, img_l, kpl, good, None, **draw_params)
    imshow(f"Draw Matches - etape {step+1}", img3)
    cv2.waitKey(0)

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kpr[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpl[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w = imgr.shape
        pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1, 1, 2)
        dst_pts_transformed = cv2.perspectiveTransform(pts, M)
        imgr_poly = cv2.polylines(imgr.copy(), [np.int32(dst_pts_transformed)], True, 255, 3, cv2.LINE_AA)
        imshow(f"right_image_overlapping - etape {step+1}", imgr_poly)
        cv2.waitKey(0)

        homographies.append(M)
        print(f"  Matches etape {step+1} : {len(good)}  => H calculee")
    else:
        print(f"Not enough matches found - {len(good)}/{MIN_MATCH_COUNT}")
        homographies.append(np.eye(3))

# ============================================================
# ETAPE 4 : Composer les homographies (cumulatif)
#
# H_cum[0] = I              => images_orig[0] reste dans son repere
# H_cum[1] = H[0]           => images_orig[1] -> repere de images_orig[0]
# H_cum[2] = H[0] @ H[1]   => images_orig[2] -> images_orig[1] -> images_orig[0]
# H_cum[k] = H[0] @ H[1] @ ... @ H[k-1]
#
# Ainsi chaque image est projetee dans le repere commun (images_orig[0])
# sans jamais matcher contre un panorama deforme.
# ============================================================
H_cum = [np.eye(3)]
for H in homographies:
    H_cum.append(H_cum[-1] @ H)

# Calculer la taille du canvas en projetant les coins de chaque image
all_corners = []
for i, img in enumerate(images_orig):
    h, w = img.shape[:2]
    corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(corners, H_cum[i])
    all_corners.append(transformed)

all_corners_np = np.concatenate(all_corners, axis=0)
x_min = int(np.floor(all_corners_np[:, :, 0].min()))
y_min = int(np.floor(all_corners_np[:, :, 1].min()))
x_max = int(np.ceil(all_corners_np[:, :, 0].max()))
y_max = int(np.ceil(all_corners_np[:, :, 1].max()))

# Translation pour que tout reste dans des coordonnees positives
T = np.array([[1, 0, -x_min],
              [0, 1, -y_min],
              [0, 0,       1]], dtype=np.float64)

canvas_w = x_max - x_min
canvas_h = y_max - y_min
print(f"\nCanvas final : {canvas_w} x {canvas_h}")

# Warper chaque image sur le canvas (droite -> gauche, pour que l'image gauche soit au-dessus)
dst = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
for i in range(len(images_orig) - 1, -1, -1):
    H_final = T @ H_cum[i]
    warped = cv2.warpPerspective(images_orig[i], H_final, (canvas_w, canvas_h))
    imshow(f"dst warpPerspective - etape {i+1}", warped)
    cv2.waitKey(0)
    # Coller l'image (meme principe que dst[0:h, 0:w] = img_l)
    mask = np.any(warped > 0, axis=2)
    dst[mask] = warped[mask]

imshow("add left image to the warped right.jpg", dst)
cv2.waitKey(0)

# ============================================================
# ETAPE 5 : Rogner les bords noirs et sauvegarder
# ============================================================
def trim(frame):
    #crop top row if empty
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom row if empty
    if not np.sum(frame[-1]):
        return trim(frame[0:-1])
    #crop left column if empty
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop right column if empty
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

#https://www.w3schools.com/python/numpy/numpy_array_slicing.asp
arr = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
print(arr[:, -1])
print(arr[:, :-2])
print(arr[:,0])
print(arr[:,1:])
print(arr[-1])
print(arr[0:-1])
print(arr[0])
print(arr[1:])

imshow("original_image_stitched_crop.jpg", trim(dst))
cv2.waitKey(0)
cv2.imwrite("original_image_stitched_crop.jpg", trim(dst))

"""