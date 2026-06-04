from picamera2 import Picamera2
import cv2
from time import sleep
import numpy as np
 
picam2 = Picamera2()
width = 640
height = 480
config = picam2.create_video_configuration(
	main = {"size": (width,height), "format": "RGB888"}
)
picam2.configure(config)
 
picam2.start()
sleep(0.5) #allows exposure to settle
picam2.set_controls({"FrameRate": 60.0})
 
def detect_hexagons(frame):
	# convert to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
	# reduce noise
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
 
	#edge detection
	edges = cv2.Canny(blurred, 50, 150)
 
	#find all contours in edge image
	contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
	#loop through all contours
	for cont in contours:
 
		#get perimeter
		peri = cv2.arcLength(cont, True)
 
		#approximate contour (change 0.04 to how close the approximation is)
		approx = cv2.approxPolyDP(cont, 0.09 * peri, True)
 
		#check if there are 6 sides
		if len(approx) == 6:
 
			#draw contour in green
			cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
 
			#label shape
			cv2.putText(frame, "HEXAGON", tuple(approx[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
 
	return frame
 
 
#Main video loop
 
 
while True:
	frame = picam2.capture_array()
 
	#frame hexagon detection
	output = detect_hexagons(frame)
 
	#display result
	#cv2.imshow("Hexagon Detection", output)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	cv2.imshow("B&W frame", gray)
 
	#'q' to quit
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
 
cv2.destroyAllWindows()