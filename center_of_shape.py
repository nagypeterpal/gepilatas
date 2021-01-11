#run command 
# python center_of_shape.py --imagedir images --colors colors.txt

# import the necessary packages
import argparse
import imutils
import cv2
import os
import numpy as np

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

    print ('---------------------------------')
    print ('current file: ' + current_file)
    print ('---------------------------------')

    #read the image file and create a copy to process
    image = cv2.imread(current_dir+'/'+current_file)
    image_clone = image.copy()
    
    #drawing predefined colors on original image
    x=1
    for row in lineList:
        col_v = row.split(',')
        cv2.rectangle(image,(x*40,0),((x+1)*40,30),(int(col_v[0]),int(col_v[1]),int(col_v[2])),-1)
        cv2.putText(image,'c_'+str(x), (x*40+5, 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)
        x+=1

    # blur the image
    blurred = cv2.GaussianBlur(image_clone, (105, 105), 0)
    #cv2.imshow("blurred", blurred)
    
    #clustering to 2 colors
    cluster_nr = 2
    #reshape to flat and convert to np.float32
    flattened_img = blurred.reshape((-1,3))
    flattened_img = np.float32(flattened_img)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    #do the kmeans
    ret,label,center=cv2.kmeans(flattened_img,cluster_nr,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    reshaped = center[label.flatten()].reshape((image.shape))
    #cv2.imshow('clustered',reshaped)

    #conv to grey for contourfinding
    gray = cv2.cvtColor(reshaped , cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray", gray)
    
    #thresholding - and inverting if needed
    thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    #invert if needed
    if (thresh[1, 1])>100: thresh = cv2.bitwise_not(thresh)
    #cv2.imshow("thresh", thresh)
    
    # find contours in the thresholded image
    cnts = cv2.findContours(thresh, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    ord_cnts= sorted(cnts, key=cv2.contourArea, reverse= True)
    
    # loop over the contours
    x=1
    for c in ord_cnts:
        
        print(' - shape ' + str(x) + '. : ')

        # compute the center of the contour
        M = cv2.moments(c)

        if M["m00"]!=0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        #compare to every color and find the closest one
        y=1
        tol_arr=[]
        b,g,r=(blurred[cY, cX])

        print('      rgb: ' +  str(r) + ' ' + str(g)+ ' ' + str(b))

        for row in lineList:
            pr_b,pr_g,pr_r = row.split(',')
            tol_g=abs(int(pr_g)-int(g))
            tol_b=abs(int(pr_b)-int(b))
            tol_r=abs(int(pr_r)-int(r))
            tol_arr.append(tol_g + tol_b + tol_r)

            y+=1

        closest_color=tol_arr.index(min(tol_arr))+1

        #draw the contours on the original image
        cv2.putText(image,'shape '+ str(x) , (cX, cY),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 2)
        cv2.putText(image,'closest color: ' + str(closest_color), (cX, cY+15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 2)
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

        if(x==1): break
                
        x+=1

    # show the image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()