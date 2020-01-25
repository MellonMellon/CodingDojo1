from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

# ---------------------------------------------------------------------------- #
#                               Argument & Setup                               #
# ---------------------------------------------------------------------------- #
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-i", "--output-img", type=str,
  help="path to output image")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# # # #
#  Load encodings faces
#
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# # # #
# Intitialize the camera
#
print("[INFO] Intitialize the camera...")
vs = VideoStream(usePiCamera=1).start()

# # # #
# Camera Warmup
#
print("[INFO] Camera warmup...")
time.sleep(2.0)


# ---------------------------------------------------------------------------- #
#                                     START                                    #
# ---------------------------------------------------------------------------- #
while True:
	# # # #
	# read current frame
	#
	frame = vs.read()

	# # # #
	# Convert BGR to RGB for openCV.
	# Resize Width in the aim to speedup processing
	#
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = imutils.resize(frame, width=250)
	r = frame.shape[1] / float(rgb.shape[1])

	# # # #
	# Get bounding boxes for each detected face
	#
	boxes = face_recognition.face_locations(rgb, model=args["detection_method"])

	# # # #
	# Encodings the founded faces.
	# This Step can be slow, it depend on ne number of faces.
	#
	encodings = face_recognition.face_encodings(rgb, boxes)


	# # # #
	# We compare each encoding faces with the encodings dataset
	#  
	names = []
	for encoding in encodings:
		# # # #
		# compare current encoding face with the dataset
		#
		matches = face_recognition.compare_faces(data["encodings"],
			encoding)
		name = "Unknown"

		# # # #
		# If matche, Get all the matched Id.
		#	Also, we maintain a count for each recognized face. each count are compare and we keep the highest
		#
		if True in matches:
	
			# Get mateched id
			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
			counts = {}

			# For each match, increment count and get name
			for i in matchedIdxs:
				name = data["names"][i]
				counts[name] = counts.get(name, 0) + 1

			# Check the highest count and get it name
			name = max(counts, key=counts.get)
		
		# Save it
		names.append(name)

	# # # #
	# Draw Boxes of all the founded faces
	#
	for ((top, right, bottom, left), name) in zip(boxes, names):
		# rescale
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)

		# draw rectangle and text
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)

 	# Save as img
	if args["output_img"] is not None:
		cv2.imwrite(args["output_img"], frame) 

# ---------------------------------------------------------------------------- #
#                                    Cleanup                                   #
# ---------------------------------------------------------------------------- #
cv2.destroyAllWindows()
vs.stop()
