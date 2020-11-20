#run command 
#run python center_of_shape.py --image shapes_and_colors.jpg --colors colors.txt

#Módosítani:
# 1. esetleg más szinter
# 2. elore definialt szinek kozul melyikhez van a legkozelebb (legyenk alapszinek)
# 3. 10 kep (életszerubb) tesztelése ember ellen
# 4. software korlatai
# 5. doksi tartalmazza a tesztelést, doksi repoba, repolink kuld véleményezésre

#------------------------------------------------------------------------------

# import the necessary packages
import argparse
import imutils
import cv2
import csv
import os

#general variables
#   threshold per channel
thres_per_ch=10

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imagedir", required=True,	help="path to the input images")
ap.add_argument("-c", "--colors", required=True, help="path to color definition")
args = vars(ap.parse_args())

directory = os.fsencode(args["imagedir"])
print ("reading directory:" + str(directory))

#read the predefined colors
list_of_colors = []
text_file = open(args["colors"], "r")
lineList = [line.rstrip('\n') for line in text_file]

#loop trough image files
for filename in os.listdir(directory):

    current_dir= directory.decode('utf-8')
    current_file= filename.decode('utf-8')

    print ("current file:" + current_file)
    print (current_file)

    #read the image file
    image = cv2.imread(current_dir+'/'+current_file)
    image_clone = image.copy()
    
    #predefined colors  drawing
    x=1
    print ("Loaded colors:")
    for row in lineList:
        col_v = row.split(',')
        #print(str(col_v[0]) +"  " + str(col_v[1]) +"  " + str(col_v[2]))
        cv2.rectangle(image,(x*40,0),((x+1)*40,30),(int(col_v[0]),int(col_v[1]),int(col_v[2])),-1)
        cv2.putText(image,'c_'+str(x), (x*40+5, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)
        x+=1

    # load the image, convert it to grayscale, blur it slightly,
    #on the copy we find the contours
    gray = cv2.cvtColor(image_clone , cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", gray)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow("blurred", blurred)
    
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("thresh", thresh)
    
    # find contours in the thresholded image
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    print ("Contours:")
    x=1
    for c in cnts:
        outtext= str(x) + '. shape in range of color: '
        # compute the center of the contour
        M = cv2.moments(c)
        if M["m00"]!=0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        #compare to every color
        y=1
        b,g,r=(image[cY, cX])
        for row in lineList:
            pr_b,pr_g,pr_r = row.split(',')
            tol_g=abs(int(pr_g)-int(g))
            tol_b=abs(int(pr_b)-int(b))
            tol_r=abs(int(pr_r)-int(r))
            if tol_g + tol_b + tol_r < thres_per_ch*3:
                outtext += ' ' + str(y)
            y+=1

        #draw
        cv2.putText(image,'s_'+str(x), (cX, cY),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 2)
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        print(outtext)
        x+=1

    # show the image
    cv2.imshow("Image", image)
    cv2.waitKey(0)


