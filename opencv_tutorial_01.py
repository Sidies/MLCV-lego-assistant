# import the necessary packages
import imutils
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help = "Input")
ap.add_argument("-o", "--output", required=False, help = "Output")
args = vars(ap.parse_args())

image = cv2.imread(args["input"])
(h, w, d) = image.shape
print(f"width={w} height={h} deepth={d}")

# display the image to our screen -- we will need to click the window
# open by OpenCV and press a key on our keyboard to continue execution
#cv2.imshow("Image", image)
#cv2.waitKey(0)

# extract a 100x100 pixel square ROI (Region of Interest) from the
# input image starting at x=360,y=30 at ending at x=460,y=130
roi = image[30:130, 360:460]
#cv2.imshow("ROI", roi)
#cv2.waitKey(0)

resized = imutils.resize(image, width=300)
#cv2.imshow("Imutils Resize", resized)
#cv2.waitKey(0)

#rotated = imutils.rotate_bound(image, 45)
#cv2.imshow("Imutils Bound Rotation", rotated)
#cv2.waitKey(0)

blurred = cv2.GaussianBlur(image, (11, 11), 0)
#cv2.imshow("Blurred", blurred)
#cv2.waitKey(0)

# draw a 2px thick red rectangle surrounding the face
output = image.copy()
cv2.rectangle(output, (360, 30), (460, 130), (0, 0, 255), 2)
#cv2.imshow("Rectangle", output)
#cv2.waitKey(0)

# draw green text on the image
#output = image.copy()
cv2.putText(output, "We got him", (10, 25), 
	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow("Text", output)
cv2.waitKey(0)