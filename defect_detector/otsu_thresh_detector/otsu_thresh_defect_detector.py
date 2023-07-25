#%% Organize Imports
import cv2
import numpy as np

#%% Main defect detection function

def detect(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect defects in an image using the otsu thresholding method.

    Parameters
    ----------
    image : np.ndarray
        The image to detect defects in.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the annotated defect image and the defect map.
    """
    # Read the image and convert to grayscale
    image_ = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # Apply Otsu's thresholding
    _, threshold_image = cv2.threshold(image_, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #apply morphological operations to remove noise
    kernel = np.ones((3,3),np.uint8)
    threshold_image = cv2.morphologyEx(threshold_image, cv2.MORPH_OPEN, kernel, iterations=2)
    threshold_image = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Invert the binary image to get the defects
    defect_map = cv2.bitwise_not(threshold_image)
    
    # blackout nonfoil region
    col_avg = np.mean(image_, axis=0)
    col_avg_normalized = (col_avg - np.min(col_avg)) / (np.max(col_avg) - np.min(col_avg))

    defect_map[:, col_avg_normalized<0.5] = 0
    

    # Find contours in the defect image
    contours, _ = cv2.findContours(defect_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Draw bounding boxes around the defects
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        contour_image = cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw a red bounding box

    return contour_image, defect_map


#%% Test the defect detector if run directly
if __name__ == "__main__":
    
    import time
    
    # Load the image
    test_image_path  = "../../data/test_image.jpg"
    test_image  = cv2.imread(test_image_path)
    
    # Run the defect detector
    t0 = time.time()
    defect_image, defect_map = detect(test_image)
    print(f"Time taken: {time.time() - t0:.4f} seconds")

    # Display the results
    cv2.imshow("Defect Image", defect_image)
    cv2.imshow("Defect Map", defect_map)

    # Wait for the user to press a key
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# %%
