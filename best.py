import cv2
import numpy as np

def trim(frame):
    # Recursive crop to remove black borders (from your reference)
    if not np.sum(frame[0]):
        return trim(frame[1:])
    if not np.sum(frame[-1]):
        return trim(frame[0:-1])
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

def stitch_two(img_l, img_r):
    # Convert to grayscale for SIFT
    imgr_gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    imgl_gray = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kpr, desr = sift.detectAndCompute(imgr_gray, None)
    kpl, desl = sift.detectAndCompute(imgl_gray, None)

    # Matching features
    match = cv2.BFMatcher()
    matches = match.knnMatch(desr, desl, k=2)

    good = []
    # Using 0.7 for ratio test (0.3 is often too strict for multi-image)
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kpr[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpl[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Find Homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Warp the right image onto the perspective of the left image
        # We expand the width to accommodate both images
        h, w = img_l.shape[:2]
        dst = cv2.warpPerspective(img_r, M, (img_l.shape[1] + img_r.shape[1], img_l.shape[0]))
        
        # Place the left image on top of the warped result
        dst[0:img_l.shape[0], 0:img_l.shape[1]] = img_l
        
        # Trim the black edges
        return trim(dst)
    else:
        print("Not enough matches found!")
        return img_l

# --- Main Execution ---
image_files = ['1111.jpg', '2222.jpg', '3333.jpg', '4444.jpg', '5555.jpg']

# Start with the first image
panorama = cv2.imread(image_files[0])

for i in range(1, len(image_files)):
    print(f"Stitching image {i+1}/{len(image_files)}...")
    next_img = cv2.imread(image_files[i])
    
    if next_img is None:
        print(f"Skipping {image_files[i]} - file not found.")
        continue
        
    # Stitch the current panorama with the next image in the list
    panorama = stitch_two(panorama, next_img)

# Save and Show Result
cv2.imwrite("final_panorama.jpg", panorama)
cv2.imshow("Final Panorama", panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()