import cv2
import numpy as np

def correct_distortion(image_path, save_path):
    # Read the distorted image
    distorted_image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(distorted_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use the keypoints detection method (e.g., using the ORB detector)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(blurred, None)

    # Get the four corner points' coordinates
    pts = np.float32([keypoint.pt for keypoint in keypoints])

    # Order the points (top-left, top-right, bottom-right, bottom-left)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Define the desired width and height of the corrected image
    width, height = 256, 256

    # Define the corresponding four points in the corrected image
    dst = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])

    # Calculate the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    corrected_image = cv2.warpPerspective(distorted_image, M, (width, height))

    # Save the corrected image in PNG format
    cv2.imwrite(save_path, corrected_image)

    # Display the original and corrected images
    cv2.imshow('Distorted Image', distorted_image)
    cv2.imshow('Corrected Image', corrected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Replace 'your_distorted_image_path.jpg' with the path to your distorted image
correct_distortion('C:/Users/User/Documents/Projects/Nilesh/RoSteALS/examples/20240118_145839.jpg', 'C:/Users/User/Documents/Projects/Nilesh/RoSteALS/examples/20240118_145839.png')