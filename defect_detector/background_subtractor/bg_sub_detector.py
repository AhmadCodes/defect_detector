# %% Organize Imports
import cv2
import numpy as np

# %%


def detect(image: np.ndarray, background: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Convert the images to grayscale
    gray_background = (
        cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        if len(background.shape) == 3
        else background.copy()
    )
    gray_current = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(image.shape) == 3
        else image.copy()
    )

    # Calculate the absolute difference between the current image and the background
    diff = cv2.absdiff(gray_current, gray_background)

    # Apply a threshold to create a binary image with defects as white and non-defects as black
    _, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    col_avg = np.mean(gray_current, axis=0)
    col_avg_normalized = (col_avg - np.min(col_avg)) / (
        np.max(col_avg) - np.min(col_avg)
    )

    thresh[:, col_avg_normalized < 0.5] = 0

    # morphological operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find the contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = (
        image.copy()
        if len(image.shape) == 3
        else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    )
    # Draw the bboxes around the contours
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append((x, y, w, h))
        cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return contour_image, thresh, bboxes


# %% Test the function if run directly

if __name__ == "__main__":
    test_image_path = "../../data/test_image.jpg"
    test_background_path = "../../data/background_image.png"

    import time

    t0 = time.time()
    defect_image, defect_mask, bboxes = detect(
        cv2.imread(test_image_path), cv2.imread(test_background_path)
    )
    print(f"Time taken: {time.time() - t0}")
    print(f"Detected {len(bboxes)} defects")

    cv2.imshow("Defect Image", defect_image)
    cv2.imshow("Defect Mask", defect_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# %%
