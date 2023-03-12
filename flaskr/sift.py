import numpy as np
import cv2 as cv
from timeit import default_timer as timer

#Minimum count of matches
MIN_MATCH_COUNT = 4

#Start timer
start = timer()

#Importing the images 
# TODO - Take in the images from the FLASK web app
img1_color = cv.imread('static/images/Image01_50pc.jpeg')
img2_color = cv.imread('static/images/Image02_50pc.jpeg')

# Gray scaling the images
img1 = cv.cvtColor(img1_color, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(img2_color, cv.COLOR_BGR2GRAY)


# Setting up the mask
contours = np.array([[0,1024], [265,590], [503,590], [768,1024]]) #w,h
image = np.zeros((1024,768), dtype='uint8') #h,w
cv.fillPoly(image, pts = [contours], color =(255,255,255))

# Applying the mask
masked1 = cv.bitwise_and(img1, img1, mask=image)
masked2 = cv.bitwise_and(img2, img2, mask=image)


# End of the timer
end = timer()

t1 = end-start

print('Time to load images: ', t1, 's', sep='')

start = timer()

# Initiate SIFT detector

# nfeatures
#   The number of best features to retain.
#   The features are ranked by their scores (measured in SIFT algorithm as the local contrast)

# nOctaveLayers
#   The number of layers in each octave. 3 is the value used in D. Lowe paper.
#   The number of octaves is computed automatically from the image resolution.

# contrastThreshold
#   The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions.
#   The larger the threshold, the fewer features are produced by the detector.

# edgeThreshold
#   The threshold used to filter out edge-like features.
#   Note that the meaning is different from the contrastThreshold, i.e. the larger the edgeThreshold,
#   the fewer features are filtered out (more features are retained).

# sigma
#   The sigma of the Gaussian applied to the input image at the octave #0.
#   If your image is captured with a weak camera with soft lenses, you might want to reduce the number.

sift = cv.SIFT_create(nfeatures=0,              # 0
                      nOctaveLayers=3,          # 3
                      contrastThreshold=0.04,   # 0.04
                      edgeThreshold=10,         # 10
                      sigma=1.6)                # 1.6

# find the key points and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(masked1, None)
kp2, des2 = sift.detectAndCompute(masked2, None)

end = timer()

t2 = end-start

print('Time to get key points: ', t2, 's', sep='')

start = timer()

FLANN_INDEX_KDTREE = 1

index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

end = timer()

t3 = end-start

print('Time to get matches: ', t3, 's', sep='')

print('\nTotal Time: ', '{:.5f}s'.format((t1+t2+t3)))


if len(good) >= MIN_MATCH_COUNT:

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 1.0)

    matchesMask = mask.ravel().tolist()

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img3 = cv.drawMatches(masked1,kp1,masked2,kp2,good,None,**draw_params)

    cv.imshow("mapped", img3)
    cv.waitKey(0)

    transformed_points = cv.perspectiveTransform(src_pts.reshape(-1, 1, 2), H)

    for f, b in zip(transformed_points.reshape(-1, 2), dst_pts):
        print(f, b)

    # Compare the transformed point with the destination points to check if it's on the plane
    if np.allclose(transformed_points.reshape(-1, 2), dst_pts, rtol=1e-3, atol=1e-3):
        print("The points are on the plane")
    else:
        print("The points are not on the plane")

else:
    print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    matchesMask = None
