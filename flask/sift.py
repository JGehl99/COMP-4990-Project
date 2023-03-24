import numpy as np
import cv2
from timeit import default_timer as timer


# Return result, execution time, and generated image from sift()
def sift_setup(img1, img2, kp):

    start = timer()
    result, image = sift(img1, img2, kp)
    end = timer()

    return result, end-start, image


def sift(img1_color, img2_color, key_points):

    # Gray scaling the images
    img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)

    height, width = img2.shape

    # Setting up the mask
    contours = np.array(
        [[0, img2.shape[0]],
         [img2.shape[1]*0.35, img2.shape[0]*0.40],
         [img2.shape[1]*0.65, img2.shape[0]*0.40],
         [img2.shape[1], img2.shape[0]]]
    )  # w,h

    image = np.zeros((height, width), dtype='uint8')  # h,w
    cv2.fillPoly(image, pts=np.int32([contours]), color=(255, 255, 255))

    # Applying the mask
    masked_img2 = cv2.bitwise_and(img2, img2, mask=image)

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
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(masked_img2, None)

    # Index type
    flann_index_kdtree = 1

    # Index and search parameters
    index_params = dict(algorithm=flann_index_kdtree, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > key_points:

        # Get list of src and dst points into the proper format to use in findHomography()
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Calculate homographic matrix M and the list of matches used to calculate it (mask)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Convert mask to list
        matchesMask = mask.ravel().tolist()

        # Height and width of image1
        h, w = img1.shape

        # Create np array the same size as img1
        transformed_points = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        # Map these points to img2 using the homographic matrix
        dst = cv2.perspectiveTransform(transformed_points, M)

        # Draw box around where the points were mapped to on img2
        img2 = cv2.polylines(masked_img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        # Draw matches between img1 and img2
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)

        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

        # Return img3 which has the matches drawn on it
        return True, img3

    else:

        # Else if there are not enough matches, concatenate the images together and return it with no matches drawn

        image_bordered = cv2.copyMakeBorder(
            src=img1,
            top=0,
            bottom=masked_img2.shape[0] - img1.shape[0],
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT
        )
        image_bordered2 = cv2.copyMakeBorder(
            image_bordered,
            0,
            0,
            0,
            50,
            cv2.BORDER_CONSTANT,
            None,
            (255, 255, 255)
        )

        img3 = np.hstack((image_bordered2, masked_img2))

        return False, img3
