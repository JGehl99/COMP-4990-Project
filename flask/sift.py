import numpy as np
import cv2
from timeit import default_timer as timer


# Return result, execution time, and generated image from sift()
def sift_setup(img1, img2, threshold):

    start = timer()
    result, image = sift(img1, img2, threshold)
    end = timer()

    if result != 2:
        return result, end-start, image
    else:
        return result, end-start, None


def sift(img1_color, img2_color, threshold):

    # Minimum count of matches
    min_match_count = 4

    # Gray scaling the images
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    # Setting up the mask
    contours = np.array([[0, 1024], [265, 590], [503, 590], [768, 1024]])  # w,h
    image = np.zeros((1024, 768), dtype='uint8')  # h,w
    cv2.fillPoly(image, pts=[contours], color=(255, 255, 255))

    # Applying the mask
    masked1 = cv2.bitwise_and(img1, img1, mask=image)
    masked2 = cv2.bitwise_and(img2, img2, mask=image)

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

    sift = cv2.SIFT_create(nfeatures=0,  # 0
                           nOctaveLayers=3,  # 3
                           contrastThreshold=0.04,  # 0.04
                           edgeThreshold=10,  # 10
                           sigma=1.6)  # 1.6

    # find the key points and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(masked1, None)
    kp2, des2 = sift.detectAndCompute(masked2, None)

    flann_index_kdtree = 1

    index_params = dict(algorithm=flann_index_kdtree, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) >= min_match_count:

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

        h, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0)

        matches_mask = mask.ravel().tolist()

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matches_mask,  # draw only inliers
                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img3 = cv2.drawMatches(masked1, kp1, masked2, kp2, good, None, **draw_params)

        transformed_points = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), h)

        # Compare the transformed point with the destination points to check if it's on the plane
        if np.allclose(transformed_points.reshape(-1, 2), dst_pts, rtol=threshold):
            return 1, img3
        else:
            return 0, img3

    else:
        return 2, None
