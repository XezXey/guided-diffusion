import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the reference image and the image to be aligned
full_image = cv2.imread('original.png')
aligned_face = cv2.imread('anakin.png')

# Convert the full image to grayscale
gray = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)

# Detect the face using dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./temp/shape_predictor_68_face_landmarks.dat')
rects = detector(gray, 1)

# Initialize dlib's face detector (HOG-based)
detector = dlib.get_frontal_face_detector()

# Detect faces in the full image
faces = detector(full_image, 1)

# Assume there's only one face in the full image
face_location = faces[0]
x, y, w, h = face_location.left(), face_location.top(), face_location.width(), face_location.height()

# Resize the aligned face to match the size of the face in the full image, if necessary
aligned_face = cv2.resize(aligned_face, (w, h))

# Create a mask for the aligned face
mask = np.zeros(full_image.shape[:2], dtype=np.uint8)
mask[y:y+h, x:x+w] = 1

# Create a copy of the full image and replace the pixels in the full image 
# that correspond to the aligned face with the pixels from the aligned face
result = np.copy(full_image)
result[mask==1] = aligned_face[mask==1]

# Use the mask to blend the aligned face with the surrounding pixels in the full image
result = cv2.seamlessClone(aligned_face, full_image, mask, (x+w//2, y+h//2), cv2.NORMAL_CLONE)

# Save the images
plt.imsave('out.png', result)
