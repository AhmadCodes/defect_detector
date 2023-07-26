# %% Organize Imports
import numpy as np
import cv2
import time
from .detector_factory import ImageDefectDetectionFactory
from .detector_factory import VideoDefectDetectionFactory
from .utils import blackout_nonfoil_region

# %%


class ImageDefectDetector:
    """
    This class is used to detect defects in an image using the specified method. Methods include: \
    edge_detector, background_subtractor, gradient_threshold_detector, otsu_thresh_detector, kalman_residual_detector, object_detector
    """

    def __init__(self, method: str = "edge_detector") -> None:
        if method not in [
            "edge_detector",
            "background_subtractor",
            "gradient_threshold_detector",
            "otsu_thresh_detector",
            "kalman_residual_detector",
            "object_detector",
        ]:
            raise ValueError("Invalid method type.")
        self.method = method
        self.defect_detection_method = (
            ImageDefectDetectionFactory.create_defect_detection_method(self.method)
        )

    def detect_defects(
        self,
        image: np.ndarray,
        background_image: np.ndarray = None,
        non_foil_blackout: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect defects in an image using the specified method.

        Parameters
        ----------
        image : np.ndarray
            The image to detect defects in.
        background_image : np.ndarray, optional
            The background image to use for background subtraction method, by default None
        non_foil_blackout : bool
            Whether to blackout the non-foil region of the image, by default True

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing the annotated defect image and the defect map.
        """

        # preprocessing if needed
        desired_height = 256
        desired_width = 3660

        if image is None:
            raise ValueError("Image must be provided.")

        image = cv2.resize(
            image, (desired_width, desired_height), interpolation=cv2.INTER_AREA
        )
        if background_image is not None:
            background_image = cv2.resize(
                background_image,
                (desired_width, desired_height),
                interpolation=cv2.INTER_AREA,
            )
        if non_foil_blackout:
            image = blackout_nonfoil_region(image)

        # Detect the defects
        t0 = time.time()
        if self.method == "background_subtractor":
            if background_image is None:
                raise ValueError(
                    "Background image must be provided for background subtraction method."
                )
            defect_image, defect_map, bboxes = self.defect_detection_method(
                image, background_image
            )
        else:
            defect_image, defect_map, bboxes = self.defect_detection_method(image)
        detection_time = time.time() - t0

        # Return the defect image and defect map
        return defect_image, defect_map, bboxes, detection_time


# %% Video Defect Detector Class
class VideoDefectDetector:
    def __init__(self, method: str = "background_subtractor") -> None:
        if method not in ["background_subtractor"]:
            raise ValueError("Invalid method type.")
        self.method = method
        self.defect_detection_method = (
            VideoDefectDetectionFactory.create_defect_detection_method(self.method)
        )

    def detect_defects(self, video: str, debug: bool = False) -> tuple[str, str, list]:
        """
        Detect defects in a video using the specified method.

        Parameters
        ----------
        video : str
            The video to detect defects in.

        Returns
        -------
        tuple[str,str,list]
            A tuple containing the path to the annotated defect video, path to the defect mask video, \
            and frames bboxes where each element in the frames_bboxes contains a list of bboxes for the given frame in that sequence.
        """

        (
            output_detection_vid,
            output_mask_vid,
            frames_bboxes_list,
        ) = self.defect_detection_method(video, debug=debug)

        return output_detection_vid, output_mask_vid, frames_bboxes_list
