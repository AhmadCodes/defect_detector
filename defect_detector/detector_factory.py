# Create a factory class to produce the desired defect detection method
class ImageDefectDetectionFactory:
    """A factory class to produce the desired defect detection method."""

    @staticmethod
    def create_defect_detection_method(method_type: str) -> callable:
        """
        Create a defect detection method based on the specified method type.

        Parameters
        ----------
        method_type : str
            The type of defect detection method to create.

        Returns
        -------
        callable
            The defect detection method.

        Raises
        ------
        ValueError
            If an invalid method type is specified.
        """

        if method_type == "edge_detector":
            from .edge_detector import detect

            return detect
        elif method_type == "background_subtractor":
            from .background_subtractor import detect

            return detect
        elif method_type == "gradient_threshold_detector":
            from .gradient_threshold_detector import detect

            return detect
        elif method_type == "otsu_thresh_detector":
            from .otsu_thresh_detector import detect

            return detect
        elif method_type == "kalman_residual_detector":
            from .kalman_residual_detector import detect

            return detect
        elif method_type == "object_detector":
            from .object_detector import detect

            return detect
        else:
            raise ValueError("Invalid method type.")


# %%


class VideoDefectDetectionFactory:
    @staticmethod
    def create_defect_detection_method(method_type: str) -> callable:
        """
        Create a defect detection method based on the specified method type.

        Parameters
        ----------
        method_type : str
            The type of defect detection method to create.

        Returns
        -------
        callable
            The defect detection method.

        Raises
        ------
        ValueError
            If an invalid method type is specified.
        """
        if method_type == "background_subtractor":
            from .video_bg_sub_detector import detect

            return detect
        else:
            raise ValueError("Invalid method type.")
