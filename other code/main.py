import cv2
import os
import numpy as np

# checking the working of SIFT (algorithm to detect, describe, and match local features in images)
filename = "SOCOFing/Altered/Altered-Hard/190__M_Left_thumb_finger_CR.BMP"
test_img = cv2.imread(filename)

best_match_score = 0
best_filename = None
best_image = None
keypt1, keypt2, matchpts = None, None, None

i = 0
for file in [file for file in os.listdir("SOCOFing/Real")][:1000]:
    if i%50 == 0:
        print("reached img ", i)
    i = i + 1    
    real_img = cv2.imread("SOCOFing/Real/" + file)
    # scale invariant feature transform
    sift = cv2.SIFT_create()    # can extract key points of an image, and descriptors of those pts
                                # we will just get key points of each image, then compare with the input image
    
    keypts_1, desc_1 = sift.detectAndCompute(test_img, None)
    keypts_2, desc_2 = sift.detectAndCompute(real_img, None)

    matches = cv2.FlannBasedMatcher({'algorithm' : 1 , 'trees' : 10}, {}).knnMatch(desc_1, desc_2, k=2)
    match_pts = []
    for p, q in matches:
        # threshold for similarity between descriptors of test_img and curr_img
        if p.distance < 0.35*q.distance: # (lowe's ratio test) 
            print("hit!")
            match_pts.append(p)
    
    keypts = np.min([len(keypts_1), len(keypts_2)])

    curr_score = len(match_pts)*100/keypts # ratio of matchpts to keypts
    if curr_score > best_match_score:
        best_match_score = curr_score
        best_filename = file
        best_image = real_img
        keypt1, keypt2, matchpts = keypts_1, keypts_2, match_pts

print("Input image: " + str(filename))
print("Best match for the image: " + str(best_filename))
print("Best score: " + str(best_match_score))

# result = cv2.drawMatches(test_img, keypt1, best_image, keypt2, matchpts, None)
# result = cv2.resize(result, None, fx=4, fy=4)
# cv2.imshow("Resultant matching: ", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


