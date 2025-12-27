import cv2
import numpy as np

img1 = cv2.imread("C:\OpenCV with python\ObjDetection\CirclesImage.jpg",cv2.IMREAD_COLOR)
grayImg = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
blurredImg = cv2.blur(grayImg,(3,3))

#HoughCircles helps in detecting circles in an image using hough gradient method
detectedCircles = cv2.HoughCircles(
    blurredImg, #Image variable
    cv2.HOUGH_GRADIENT, #Circle detection method
    1, #1 will give resolution of the image as the input image (2 will give half the resolution)
    20, 
    param1 = 50, #Canny edge detector
    param2 = 30,
    minRadius= 50, 
    maxRadius= 80
)

if detectedCircles is not None:
    detectedCircles = np.uint16(np.around(detectedCircles))
    for i in detectedCircles[0,:]:
        a,b,r = i[0],i[1],i[2] #a is x coord of the center, b is y coord of the center, r is radius of the circle
        cv2.circle(img1,(a,b),r,(0,0,0))
cv2.imshow("Detected Circles", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Simple blob detector
img2 = cv2.imread("C:\OpenCV with python\ObjDetection\images.jpeg",0)

#Simple blob detector regions (blobs) based on area, circularity, convexity and inertia
params = cv2.SimpleBlobDetector_Params()#Object to store filtering parameters
params.filterByArea = True
params.filterByColor = True
params.blobColor = 255
params.minArea = 1500
params.maxArea = 15000
params.filterByCircularity = True
params.minCircularity = 0.6
params.filterByConvexity = False
params.minConvexity = 0.2
params.filterByInertia = False
params.minInertiaRatio = 0.01

blobDet = cv2.SimpleBlobDetector_create(params)
keypoints = blobDet.detect(img2)

blank = np.zeros((1,1))
kp = cv2.drawKeypoints(
    img2,
    keypoints,
    blank,
    (255,0,0),
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)
number = len(keypoints)
text = "Number of blobs: " + str(number)
cv2.putText(kp,text,(0,20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,(125,255,0),2)
cv2.imshow("Blob detector",kp)
cv2.waitKey(0)
cv2.destroyAllWindows()
