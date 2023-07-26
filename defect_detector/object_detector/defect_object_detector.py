# %%
import cv2
import ultralytics
import numpy as np
import os

# %%
model_path = os.path.join(
    os.path.dirname(__file__), "../../weights/yolov8n/checkpoint.pt"
)
print(model_path)
model = ultralytics.YOLO(model_path)
# %%


def detect(
    image: np.ndarray, confidence_threshold: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect defects in an image using the trained defect detector (yolov8n)

    Parameters
    ----------
    image : np.ndarray
        The image to detect defects in.
    confidence_threshold : float, optional
        The threshold for the object detector, by default 0.2

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the annotated defect image and the defect map.
    """

    # Convert the image to RGB if not already
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image

    # Run the defect detector
    results = model([image], conf=confidence_threshold)

    # If no defects are found, return the original image
    if len(results) == 0:
        return image

    # Create a copy of the image to draw the results on
    defect_image = image.copy()
    defect_map = np.zeros_like(image)

    # Overlay the results on the image
    for result in results:
        # Get the bounding box and class of the defect
        box = result.boxes.xyxy[0].cpu().numpy()
        box = box.astype(int)
        cls = int(result.boxes.cls.cpu().numpy())
        if cls == 0:
            color = (0, 0, 255)
        # Overlay the bounding box on the image
        defect_image = cv2.rectangle(
            defect_image, (box[0], box[1]), (box[2], box[3]), color, 2
        )
        # Overlay the bounding region on the defect map
        defect_map = cv2.rectangle(
            defect_map, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), -1
        )
    return defect_image, defect_map


# %%
if __name__ == "__main__":
    import time

    # Load the image
    test_image_path = "../../data/test_image.jpg"
    test_image = cv2.imread(test_image_path)

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
