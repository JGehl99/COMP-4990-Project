from __future__ import print_function
import cv2 as cv
import numpy as np
from timeit import default_timer as timer

def sift_flann():
    img1 = cv.imread('images_test/1_50.jpg', cv.IMREAD_GRAYSCALE)# queryImage
    img2 = cv.imread('flask/static/images_test/2_50.jpg', cv.IMREAD_GRAYSCALE) # trainImage
    #-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors

    detector = cv.SIFT_create()

    keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
    #-- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
    #-- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.7
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    #-- Draw matches
    img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
    print(len(good_matches))
    cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # img3 = cv.resize(img_matches, (1200, 750))
    return img_matches

def orb_brute():
    img1 = cv.imread('flask/static/images_test/1_50.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
    img2 = cv.imread('flask/static/images_test/2_50.jpg',cv.IMREAD_GRAYSCALE) # trainImage
    # Initiate ORB detector
    orb = cv.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    print(len(matches))
    img3 = cv.drawMatches(img1,kp1,img2,kp2,matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # img3 = cv.resize(img3, (1200, 750))
    return img3
    # plt.imshow(img3),plt.show()

# run tests for time.    
start = timer()
# result = sift_flann()
result = orb_brute()
end = timer()

time = end-start
print(time)

# cv.putText(result, 'Time='+str(end-start), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2, 1)
cv.imwrite('sift_flann_50.jpg', result)

# cv.imshow('Good Matches', result)
# cv.waitKey()