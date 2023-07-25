#%% Organize Imports 
import cv2
import numpy as np

#%% Detector function

def detect(image: np.ndarray) -> tuple[np.ndarray, np.ndarray] :
    """
    Detect the defects in an image using edges 

    Parameters
    ----------
    image : np.ndarray
        The image to detect defects in.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the image with the detected defects and the edges of the image.
    """
    
    # Convert the grayscale image to rgb color if not already
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if image.ndim == 2 else image.copy()

    # Remove small noise from the image
    blurred_image = cv2.GaussianBlur(image_rgb, (3, 3), 0)

    # Perform edge detection using the Canny algorithm
    canny_edges = cv2.Canny(blurred_image, 20, 150)
    
    col_avg = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim==3 else image,
                      axis=0)
    col_avg_normalized = (col_avg - np.min(col_avg)) / (np.max(col_avg) - np.min(col_avg))

    canny_edges[:, col_avg_normalized<0.5] = 0

    
    # Find contours in the edge image
    contours, _ = cv2.findContours(canny_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the image for drawing contours
    contour_image = image_rgb.copy()

    # Iterate over the contours and draw them on the contour image
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Draw the bounding box on the image
        cv2.rectangle(contour_image, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red color, thickness 2

    return contour_image, canny_edges

#%%
if __name__ == "__main__":
    
    import time
    
    # load test images
    test_image_path = "../../data/test_image.jpg"
    test_image = cv2.imread(test_image_path)

    # Run defect detector
    t0 = time.time()
    defect_contours, defect_edges = detect(test_image)
    print(f"Execution Time: {time.time() - t0}s")
    # Show the images
    cv2.imshow("Detected Defects", defect_contours)
    cv2.imshow("Detected Edges", defect_edges)
    
    # View image and cleanup
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# %%
