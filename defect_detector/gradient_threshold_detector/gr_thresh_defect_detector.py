# %% Organize imports
import cv2
import numpy as np

# %% Define detector function


def detect(
    image: np.ndarray, threshold: int = 20, defect_area_threshold: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect defects in an image using vertical gradient thresholding.

    Parameters
    ----------
    image : np.ndarray
        image to detect defects in.
    threshold : int, optional
        Value that thresholds the amount of gradient to be considered as defect, by default 10
    defect_area_threshold : int, optional
        Minimum area of a defect to be considered as a defect, by default 50

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple containing the defect image and the binary image of thresholded vertical gradients.
    """

    # Convert to grayscale if not already
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    # smoothen the image
    defect_img = cv2.blur(image, (3, 3))

    # apply histogram equalization
    # defect_img = cv2.equalizeHist(defect_img)

    # Compute the gradient along the y-axis (rows)
    gradient_y = cv2.Sobel(defect_img, cv2.CV_64F, 0, 1, ksize=3)

    # Apply thresholding to obtain a binary image
    threshold_value = threshold  # Adjust this threshold value as needed
    _, binary_image = cv2.threshold(
        np.abs(gradient_y).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Convert the binary image to the appropriate format
    binary_image = binary_image.astype(np.uint8)

    # Perform morphological operations to enhance or remove noise
    kernel = np.ones((3, 3), np.uint8)
    ## apply closing
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    col_avg = np.mean(image, axis=0)
    col_avg_normalized = (col_avg - np.min(col_avg)) / (
        np.max(col_avg) - np.min(col_avg)
    )

    binary_image[:, col_avg_normalized < 0.5] = 0

    # Find contours in the binary image
    contours, _ = cv2.findContours(
        binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Analyze the contours to detect and locate defects
    defects = []
    min_defect_area = (
        defect_area_threshold  # Adjust this minimum area threshold as needed
    )

    # Draw the contours on the original image
    contour_image = (
        image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    )
    bboxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_defect_area:
            x, y, w, h = cv2.boundingRect(contour)
            bboxes.append((x, y, w, h))
            defects.append((x, y, w, h))
            # draw the rectangle on the defect contours on the original image
            cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return contour_image, binary_image, bboxes


# %% Test the detector if this file is run directly

if __name__ == "__main__":
    import time

    # Load the test image
    test_image = cv2.imread("../../data/test_image.jpg")

    t0 = time.time()
    # Detect defects in the test image
    defect_image, binary_image, bboxes = detect(test_image)
    print(f"Time taken: {time.time() - t0:.4f} seconds")
    print(f"Detected {len(bboxes)} defects")

    # Show the images
    cv2.imshow("Detected defects", defect_image)
    cv2.imshow("Binary image", binary_image)

    # Wait for key press and cleanup
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# %%
