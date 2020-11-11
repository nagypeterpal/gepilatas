# import the necessary packages
import argparse
import imutils
import cv2
import csv

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,	help="path to the input image")
ap.add_argument("-c", "--colors", required=True, help="path to color definition")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image_clone = image.copy()

#first we read the predefined colors and draw them
list_of_colors = []
text_file = open(args["colors"], "r")
lineList = [line.rstrip('\n') for line in text_file]
x=10
print ("Loaded colors:")
for row in lineList:
    col_v = row.split(',')
    print(str(col_v[0]) +"  " + str(col_v[1]) +"  " + str(col_v[2]))
    cv2.rectangle(image,(x,0),(x+20,20),(int(col_v[0]),int(col_v[1]),int(col_v[2])),-1)
    x+=20

# load the image, convert it to grayscale, blur it slightly,
#on the copy we find the contours
gray = cv2.cvtColor(image_clone , cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
# find contours in the thresholded image
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# loop over the contours
print ("Contours:")
x=1
for c in cnts:
	# compute the center of the contour
    M = cv2.moments(c)
    if M["m00"]!=0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
	# draw the contour and center of the shape on the image
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(image,str(x), (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    print(str(x) + ". color: " + str(image[cX,cY,0]) + " - " + str(image[cX,cY,1]) + " - " + str(image[cX,cY,2]))
	# show the image
    cv2.imshow("Image", image)
    x+=1
cv2.waitKey(0)
#run python center_of_shape.py --image shapes_and_colors.jpg --colors colors.txt
