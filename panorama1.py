import cv2
import numpy as np
import tkinter as tk 
from tkinter import filedialog
import os

# ============================================================
# Affichage avec redimensionnement automatique
# ============================================================

MAX_DISPLAY_W = 1280
MAX_DISPLAY_H = 720

def imshow(title, img):
    h, w = img.shape[:2]
    scale = min(MAX_DISPLAY_W / w, MAX_DISPLAY_H / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imshow(title, img)

# Adapter les keypoints detectes sur image reduite
def scale_keypoints(kps, scale=2.0): 
    scaled = []
    for kp in kps:
        kp2 = cv2.KeyPoint(
            kp.pt[0] * scale,
            kp.pt[1] * scale,
            kp.size * scale,
            kp.angle,
            kp.response,
            kp.octave,
            kp.class_id
        )
        scaled.append(kp2)
    return scaled

# ============================================================
# Selection des images depuis le PC
# ============================================================

root = tk.Tk()
root.withdraw()

chemins = filedialog.askopenfilenames(
    title="Selectionnez 2 a 5 images",
    filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff")]
)

root.destroy()

if len(chemins) < 2 or len(chemins) > 5:
    print("Selectionnez entre 2 et 5 images")
    exit()

# Lecture image compatible avec chemins speciaux
def imread_unicode(path):
    stream = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(stream, cv2.IMREAD_COLOR)

# ============================================================
# Charger les images
# ============================================================

images = []
image_files = []

for c in chemins:
    img = imread_unicode(c)
    if img is not None:
        images.append(img)
        image_files.append(os.path.basename(c))
        print("Chargee :", os.path.basename(c))

print("\nNombre d'images :", len(images))

# ============================================================
# Detection SIFT
# ============================================================

sift = cv2.SIFT_create() 

kps = []   # keypoints
dess = []  # descripteurs

for img in images:
    gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0,0), fx=0.5, fy=0.5)
    kp, des = sift.detectAndCompute(gray, None)
    kps.append(kp)
    dess.append(des)

# ============================================================
# Detection automatique de l'ordre gauche -> droite
# CORRECTION : seuil Lowe passe de 0.3 a 0.75
# ============================================================

n = len(images) 
scores = np.zeros(n)

matcher_order = cv2.BFMatcher() 

for i in range(n): 
    for j in range(n):
        if i == j:
            continue 

        matches_ij = matcher_order.knnMatch(dess[i], dess[j], k=2)

        # CORRECTION : 0.3 -> 0.75 pour ne pas rater des correspondances valides
        good_ij = [m for m, nn in matches_ij if m.distance < 0.75 * nn.distance]

        if len(good_ij) < 10:
            continue

        src_pts = np.float32([kps[i][m.queryIdx].pt for m in good_ij]).reshape(-1,1,2) 
        dst_pts = np.float32([kps[j][m.trainIdx].pt for m in good_ij]).reshape(-1,1,2)

        M_ij, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) 

        if M_ij is not None:
            scores[i] += M_ij[0,2] 

order = np.argsort(scores)

images_orig = [images[k] for k in order]
names_orig  = [image_files[k] for k in order]

print("\nOrdre detecte (gauche -> droite) :")
for rank,(name,k) in enumerate(zip(names_orig,order)): 
    print(rank+1,".",name,"score =",scores[k])

# ============================================================
# Fonction pour supprimer les bords noirs
# ============================================================

def trim(frame):
    while frame.shape[0] > 0 and not np.sum(frame[0]):
        frame = frame[1:]
    while frame.shape[0] > 0 and not np.sum(frame[-1]):
        frame = frame[:-1]
    while frame.shape[1] > 0 and not np.sum(frame[:,0]):
        frame = frame[:,1:]
    while frame.shape[1] > 0 and not np.sum(frame[:,-1]):
        frame = frame[:,:-1]
    return frame

# ============================================================
# CORRECTION : Fusion avec canvas dynamique
# Calcule la taille reelle necessaire pour accueillir chaque image
# ============================================================

def fusionner(panorama, img_r, M):
    """
    Fusionne img_r dans panorama via la matrice d'homographie M.
    Le canvas est calcule dynamiquement pour ne rien couper.
    """
    h_r, w_r = img_r.shape[:2]
    h_p, w_p = panorama.shape[:2]

    # Coins de l'image a assembler apres transformation
    corners_r = np.float32([[0,0],[w_r,0],[w_r,h_r],[0,h_r]]).reshape(-1,1,2)
    corners_r_transformed = cv2.perspectiveTransform(corners_r, M)

    # Coins du panorama actuel (pas de transformation, deja en place)
    corners_p = np.float32([[0,0],[w_p,0],[w_p,h_p],[0,h_p]]).reshape(-1,1,2)

    # Tous les coins reunis pour trouver les bornes du canvas final
    all_corners = np.concatenate([corners_r_transformed, corners_p], axis=0)

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Matrice de translation pour ramener tout en coordonnees positives
    T = np.array([[1, 0, -x_min],
                  [0, 1, -y_min],
                  [0, 0,      1]], dtype=np.float64)

    # Largeur et hauteur du canvas final
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min

    print(f"  Canvas final : {canvas_w} x {canvas_h} px")

    # Projeter img_r dans le canvas avec la translation
    dst = cv2.warpPerspective(img_r, T @ M, (canvas_w, canvas_h))

    # Copier le panorama actuel dans le canvas (avec decalage si y_min ou x_min < 0)
    y_off = -y_min
    x_off = -x_min
    dst[y_off:y_off+h_p, x_off:x_off+w_p] = panorama

    return dst

# ============================================================
# Assemblage iteratif des images
# ============================================================

MIN_MATCH_COUNT = 10 

panorama  = images_orig[0] 
remaining = list(zip(names_orig[1:], images_orig[1:]))

step = 1

while len(remaining) > 0:

    print("\n=== Etape", step, "===")

    gray_pano = cv2.resize(cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY),(0,0),fx=0.5,fy=0.5) 
    kp_pano, des_pano = sift.detectAndCompute(gray_pano, None)

    best_idx  = 0
    best_good = []  

    for idx,(name_r,img_r) in enumerate(remaining):
        gray_r = cv2.resize(cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY),(0,0),fx=0.5,fy=0.5) 
        kp_r, des_r = sift.detectAndCompute(gray_r, None)

        match2   = cv2.BFMatcher()
        matches2 = match2.knnMatch(des_r, des_pano, k=2) 

        good = [m for m,nn in matches2 if m.distance < 0.75*nn.distance]

        print(name_r, "->", len(good), "matches") 

        if len(good) > len(best_good): 
            best_good = good
            best_idx  = idx

    best_name, img_r = remaining[best_idx]
    print("Meilleure image :", best_name) 

    imgr = cv2.resize(cv2.cvtColor(img_r,  cv2.COLOR_BGR2GRAY),(0,0),fx=0.5,fy=0.5) 
    imgl = cv2.resize(cv2.cvtColor(panorama,cv2.COLOR_BGR2GRAY),(0,0),fx=0.5,fy=0.5)

    kpr, desr = sift.detectAndCompute(imgr, None)
    kpl, desl = sift.detectAndCompute(imgl, None) 

    # Affichage keypoints
    imshow("left keypoints",  cv2.drawKeypoints(panorama, scale_keypoints(kpl), None))
    cv2.waitKey(0)
    imshow("right keypoints", cv2.drawKeypoints(img_r,    scale_keypoints(kpr), None))
    cv2.waitKey(0)

    match3   = cv2.BFMatcher() 
    matches3 = match3.knnMatch(desr, desl, k=2)

    good = [m for m,nn in matches3 if m.distance < 0.75*nn.distance]

    draw_params = dict(matchColor=(0,255,0), singlePointColor=None, flags=2)
    img3 = cv2.drawMatches(img_r, scale_keypoints(kpr),
                           panorama, scale_keypoints(kpl),
                           good, None, **draw_params)
    imshow("Draw Matches", img3)
    cv2.waitKey(0)

    if len(good) > MIN_MATCH_COUNT:

        # Ramener les points en coordonnees pleine resolution (x2 car on a reduit a 0.5)
        src_pts = np.float32([kpr[m.queryIdx].pt for m in good]).reshape(-1,1,2) * 2 
        dst_pts = np.float32([kpl[m.trainIdx].pt for m in good]).reshape(-1,1,2) * 2

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is None:
            print("Homographie echouee, image ignoree")
            remaining.pop(best_idx)
            step += 1
            continue

        # CORRECTION : fusion avec canvas dynamique
        panorama = fusionner(panorama, img_r, M)

        imshow("Panorama etape " + str(step), panorama)
        cv2.waitKey(0)

    else:
        print("Pas assez de matches :", len(good), "< ", MIN_MATCH_COUNT)

    remaining.pop(best_idx) 
    step += 1 

# ============================================================
# Panorama final
# ============================================================

result = trim(panorama)

imshow("Panorama final", result)
cv2.waitKey(0) 

cv2.imwrite("panorama_result.jpg", result)
print("\nPanorama sauvegarde : panorama_result.jpg")