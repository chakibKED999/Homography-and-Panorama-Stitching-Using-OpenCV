import cv2
import numpy as np
import sys

# ==============================
# Mouse Click Handler
# ==============================
def mouseHandler(event,x,y,flags,param):
    global pts, im_temp

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pts) < 4:
            pts.append([x,y])
            cv2.circle(im_temp,(x,y),5,(0,255,255),-1)
            cv2.imshow("Image", im_temp)

# ==============================
# MODE 1 : REDRESSEMENT
# ==============================
def rectify(image):

    global pts, im_temp
    pts = []

    im_temp = image.copy()

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouseHandler)

    print("Cliquez 4 points dans le sens horaire")

    while True:
        cv2.imshow("Image", im_temp)
        key = cv2.waitKey(1)
        if len(pts) == 4:
            break

    pts_src = np.array(pts, dtype="float32")

    width, height = 600, 400
    pts_dst = np.array([
        [0,0],
        [width-1,0],
        [width-1,height-1],
        [0,height-1]
    ], dtype="float32")

    H, _ = cv2.findHomography(pts_src, pts_dst)
    result = cv2.warpPerspective(image, H, (width,height))

    cv2.imshow("Rectified", result)
    cv2.imwrite("rectified.png", result)
    cv2.waitKey(0)

# ==============================
# MODE 2 : INSERTION
# ==============================
def insert(src, dst):

    global pts, im_temp
    pts = []

    im_temp = dst.copy()

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouseHandler)

    print("Cliquez 4 points destination")

    while True:
        cv2.imshow("Image", im_temp)
        key = cv2.waitKey(1)
        if len(pts) == 4:
            break

    h, w = src.shape[:2]

    pts_src = np.array([
        [0,0],
        [w-1,0],
        [w-1,h-1],
        [0,h-1]
    ], dtype="float32")

    pts_dst = np.array(pts, dtype="float32")

    H, _ = cv2.findHomography(pts_src, pts_dst)

    warped = cv2.warpPerspective(src, H, (dst.shape[1], dst.shape[0]))

    mask = np.zeros(dst.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts_dst.astype(int), (255,255,255))

    dst = cv2.bitwise_and(dst, cv2.bitwise_not(mask))
    result = cv2.add(dst, warped)

    cv2.imshow("Inserted", result)
    cv2.imwrite("inserted.png", result)
    cv2.waitKey(0)

# ==============================
# MAIN
# ==============================
"""mode = input("Choisir mode: 1=Redresser  2=Insertion : ")

if mode == "1":
    img = cv2.imread("book1.jpg")
    rectify(img)

elif mode == "2":
    src = cv2.imread("source.jpg")
    dst = cv2.imread("destination.jpg")
    insert(src, dst)"""

mode = input("Choisir mode: 1=Redresser  2=Insertion : ")

if mode == "1":
    path = input("Donner le chemin de l'image à redresser: ")
    img = cv2.imread(path)

    if img is None:
        print("Erreur: image non trouvée")
        exit()

    rectify(img)

elif mode == "2":
    src_path = input("Donner le chemin de l'image source: ")
    dst_path = input("Donner le chemin de l'image destination: ")

    src = cv2.imread(src_path)
    dst = cv2.imread(dst_path)

    if src is None or dst is None:
        print("Erreur: image non trouvée")
        exit()

    insert(src, dst)