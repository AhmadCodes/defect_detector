import gradio as gr
import cv2
import numpy as np
from defect_detector import ImageDefectDetector

def image_mod(image: np.ndarray,
              method: str,
              bg_image: np.ndarray | None = None,
              rem_bg: bool = False
              ) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    image : np.ndarray
        Input image as a NumPy array.
    method : str
        Defect detection method to use.
    bg_image : np.ndarray, optional
        Background image as a NumPy array, by default None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the defect image and defect map as NumPy arrays.
    """
    print(f"Inputs: \n\
        image.shape = {image.shape}, \n\
        method = {method}, \n\
        bg_image.shape = {bg_image.shape if bg_image is not None else None}, \n\
        rem_bg = {rem_bg}")
    defect_detector = ImageDefectDetector(method=method)
    defect_image, defect_map, _ = defect_detector.detect_defects(image, background_image=bg_image, non_foil_blackout=rem_bg)
    return defect_image[...,::-1], defect_map

methods = ["edge_detector", 
            "background_subtractor",
            "gradient_threshold_detector", 
            "otsu_thresh_detector", 
            "kalman_residual_detector",
            "object_detector"]

demo = gr.Interface(
    fn=image_mod,
    inputs=[gr.Image(type="numpy", label="Input Image"), 
            gr.Dropdown(methods, label="Method"),
            gr.Image(type="numpy", label="Background Image"),
            gr.Checkbox( label="Blackout non-foil region")],
    outputs=[gr.Image(type="numpy",label="Defect Image"), 
             gr.Image(type="numpy",label="Defect Map")],
    flagging_options=["incorrect", "correct"],
)

if __name__ == "__main__":
    demo.launch(debug=True)