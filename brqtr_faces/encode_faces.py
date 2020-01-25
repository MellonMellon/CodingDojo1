from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# ---------------------------------------------------------------------------- #
#                               Argument & Setup                               #
# ---------------------------------------------------------------------------- #
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# # # #
# Get path for all images in dataset folder
#
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

knownEncodings = []
knownNames = []

# ---------------------------------------------------------------------------- #
#                                     START                                    #
# ---------------------------------------------------------------------------- #
for (i, imagePath) in enumerate(imagePaths):
	# # # #
	# Get name for the current image. (the name is the folder name)
	# 
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# # # #
	# Load image and convert it to RGB
	#
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# # # #
	# Get bounding boxes for each detected face
	#
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])

	# # # #
	# Encodings the founded faces.
	# This Step can be slow, it depend on ne number of faces.
	#
	encodings = face_recognition.face_encodings(rgb, boxes)

	# # # #
	# For each encoding image, prepare the couple encoding + name
	#
	for encoding in encodings:
		# add each encoding + name to our set of known names and
		# encodings
		knownEncodings.append(encoding)
		knownNames.append(name)

# # # #
# Save it all
# 
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()