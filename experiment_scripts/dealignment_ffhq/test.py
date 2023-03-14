import numpy as np
import cv2

# Define the values of transform_size, quad, and img
transform_size = 256
quad = np.array([[0, 0], [0, 100], [100, 100], [100, 0]])
img = np.zeros((512, 512))

# Define the values of after_transf_coor and inv_quad
after_transf_coor = np.array([[0, 0], [0, transform_size], [transform_size, transform_size], [transform_size, 0]], dtype=np.float32)
inv_quad = cv2.getPerspectiveTransform(np.array(quad, dtype=np.float32), np.array(after_transf_coor, dtype=np.float32))

# Add a column of ones to after_transf_coor to represent homogeneous coordinates
after_transf_coor_homog = np.concatenate([after_transf_coor, np.ones((4, 1))], axis=1)

# Compute the corresponding points in the original image using inv_quad
corresponding_points_homog = np.matmul(after_transf_coor_homog, inv_quad.T)

# Divide by the last coordinate to get the corresponding points in Euclidean coordinates
corresponding_points = corresponding_points_homog[:, :2] / corresponding_points_homog[:, 2:]

print(corresponding_points, quad)
# Check if corresponding_points is equal to quad
print(np.allclose(corresponding_points, quad))
