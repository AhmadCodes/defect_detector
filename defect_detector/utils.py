# %%
import cv2
import numpy as np

# %%


def blackout_nonfoil_region(image: np.ndarray) -> np.ndarray:
    """
    Blackout the non-foil region of the image.

    Parameters
    ----------
    image : np.ndarray
        The image to blackout the non-foil region of.

    Returns
    -------
    np.ndarray
        The image with the non-foil region blacked out.
    """

    image_ = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    )

    col_avg = np.mean(image_, axis=0)
    col_avg_normalized = (col_avg - np.min(col_avg)) / (
        np.max(col_avg) - np.min(col_avg)
    )

    # Find the crossing indices using vectorized operations
    image_[:, col_avg_normalized < 0.5] = 0
    return image_


# %%
if __name__ == "__main__":
    test_image_path = "../data/test_image.jpg"
    test_image = cv2.imread(test_image_path)

    test_image_ = blackout_nonfoil_region(test_image)

    cv2.imshow("test_image_", test_image_)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
# %%
