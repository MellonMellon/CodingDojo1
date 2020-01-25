from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import warnings
import datetime
import imutils
import json
import time
import cv2

# ---------------------------------------------------------------------------- #
#                               Argument & Parser                              #
# ---------------------------------------------------------------------------- #
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
	help="path to the JSON configuration file")
args = vars(ap.parse_args())

# filter warnings, load the configuration
warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))


# ---------------------------------------------------------------------------- #
#                                 Setup Camera                                 #
# ---------------------------------------------------------------------------- #
camera = PiCamera()
camera.resolution = tuple(conf["resolution"])
camera.framerate = conf["fps"]
rawCapture = PiRGBArray(camera, size=tuple(conf["resolution"]))


# ---------------------------------------------------------------------------- #
#                                 Camera Warmup                                #
# ---------------------------------------------------------------------------- #
print("[INFO] warming up...")
time.sleep(conf["camera_warmup_time"])
avg = None
lastUploaded = datetime.datetime.now()
motionCounter = 0


# ---------------------------------------------------------------------------- #
#                                     START                                    #
# ---------------------------------------------------------------------------- #
for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	frame = f.array
	timestamp = datetime.datetime.now()
	text = "Unoccupied"

	# # # #
	# Reduce size for speedup processing and use grayscale + GaussianBlur for reducing image noise
	#
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)

	# # # #
	# if the average frame is None, initialize it
	#
	if avg is None:
		print("[INFO] starting background model...")
		avg = gray.copy().astype("float")
		rawCapture.truncate(0)
		continue

	# # # # #
	# Here, we want to isolate the difference between the previous frame and the current.
	#
	cv2.accumulateWeighted(gray, avg, 0.5)
	frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

	# # # #
	# We can now Threshold the delta image.
	# Contours are easier to find by dilating the thresholded image.
	# Threshold => make black and white difference
	# dilatation => increase white zone
	#
	thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255,
		cv2.THRESH_BINARY)[1]
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# # # #
	# Draw all Contours
	#
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < conf["min_area"]:
			continue

		# # # #
		# Draw the bounding box of a contour in the frame
		#
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		
	# # # #
	# Save the current frame
	#
	if conf["show_img"]:
		cv2.imwrite(conf["show_img"], frame) 
	
	# # # #
	# clear the stream in preparation for the next frame
	#
	rawCapture.truncate(0)
