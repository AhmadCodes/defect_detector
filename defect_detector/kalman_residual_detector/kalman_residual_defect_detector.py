# %% Organize imports
import cv2
import numpy as np

# %% Helper Functions


def run_kalman_filter(
    image: np.ndarray,
    process_noise_cov: float = 1e-5,
    measurement_noise_cov: float = 1e-1,
) -> np.ndarray:
    """
    Run Kalman filter on columns of an image.

    Parameters
    ----------
    image : np.ndarray
        Image to run the Kalman filter on.
    proc_noise_cov : float, optional
        Process noise covariance, by default 1e-4
    meas_noise_cov : float, optional
        Measurement noise covariance, by default 1e-1

    Returns
    -------
    np.ndarray
        Residuals of the Kalman filter.
    """

    # Convert the image to grayscale
    gray_image = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if len(image.shape) == 3
        else image.copy()
    )
    gray_image = gray_image.astype(np.float32)  # Convert to float32

    # Define Kalman filter parameters
    state_dim = 1  # Dimension of the state vector (1D column intensity)
    measurement_dim = 1  # Dimension of the measurement vector (1D column intensity)
    transition_matrix = np.eye(
        state_dim, dtype=np.float32
    )  # Identity matrix as the state transition matrix
    observation_matrix = np.eye(
        measurement_dim, state_dim, dtype=np.float32
    )  # Identity matrix as the observation matrix
    process_noise_cov_matrix = (
        np.eye(state_dim, dtype=np.float32) * process_noise_cov
    )  # Process noise covariance matrix
    measurement_noise_cov_matrix = (
        np.eye(measurement_dim, dtype=np.float32) * measurement_noise_cov
    )  # Measurement noise covariance matrix

    # Initialize the Kalman filter for each column
    kalman_filters = []
    for col in range(gray_image.shape[1]):
        kalman_filter = cv2.KalmanFilter(state_dim, measurement_dim)
        kalman_filter.transitionMatrix = transition_matrix
        kalman_filter.measurementMatrix = observation_matrix
        kalman_filter.processNoiseCov = process_noise_cov_matrix
        kalman_filter.measurementNoiseCov = measurement_noise_cov_matrix
        kalman_filter.statePost = np.array(
            [[gray_image[0, col]]], dtype=np.float32
        )  # Initialize state
        kalman_filter.errorCovPost = np.eye(
            state_dim, dtype=np.float32
        )  # Initialize state covariance
        kalman_filters.append(kalman_filter)

    # Iterate over each column and run the Kalman filter
    residuals = np.zeros_like(gray_image, dtype=np.float32)
    for col in range(gray_image.shape[1]):
        for row in range(1, gray_image.shape[0]):
            # Predict the next state based on the transition matrix
            predicted_state = kalman_filters[col].predict()

            # Update the state based on the observed measurement
            observed_measurement = np.array([[gray_image[row, col]]], dtype=np.float32)
            corrected_state = kalman_filters[col].correct(observed_measurement)

            # Calculate the residual (difference between observed and predicted measurement)
            residual = observed_measurement[0, 0] - predicted_state[0, 0]
            residuals[row, col] = residual

    return residuals


def map_defects(residuals: np.ndarray) -> np.ndarray:
    """
    Threshold the defects based on the residuals of the Kalman filter.

    Parameters
    ----------
    residuals : _type_
        residuals of the Kalman filter.

    Returns
    -------
    np.ndarray
        Binary image of the thresholded residuals.
    """

    # apply otsu thresholding
    _, binary = cv2.threshold(
        residuals.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # apply morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary


# %% Main defect detector function
def detect(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect defects in an image using the Kalman residual defect detector

    Parameters
    ----------
    image : np.ndarray
        The image to detect defects in.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the annotated defect image and the defect map.
    """

    # Run the Kalman filter on the columns of the image
    residuals = run_kalman_filter(image)

    col_avg = np.mean(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image, axis=0
    )
    col_avg_normalized = (col_avg - np.min(col_avg)) / (
        np.max(col_avg) - np.min(col_avg)
    )

    residuals[:, col_avg_normalized < 0.5] = 0

    # Detect defects based on the residuals
    defect_image = map_defects(residuals).astype(np.uint8)

    # Find contours in the binary image
    contours, _ = cv2.findContours(
        defect_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Create a copy of the image for drawing bounding boxes
    bbox_image = (
        image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    )
    bboxes = []
    # Iterate over the contours and draw bounding boxes
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append((x, y, w, h))
        cv2.rectangle(bbox_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return bbox_image, defect_image, bboxes


# %%
if __name__ == "__main__":
    import time

    # Load the image
    test_image_path = "../../data/test_image.jpg"
    test_image = cv2.imread(test_image_path)

    # Run the defect detector
    t0 = time.time()
    defect_image, defect_map, bboxes = detect(test_image)
    print(f"Time taken: {time.time() - t0:.4f} seconds")
    print(f"Detected {len(bboxes)} defects")
    # Display the results
    cv2.imshow("Defect Image", defect_image)
    cv2.imshow("Defect Map", defect_map)

    # Wait for key press and cleanup
    cv2.waitKey(0)
    cv2.destroyAllWindows()
