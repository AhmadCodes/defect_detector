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
    defect_image, defect_map, timetaken = defect_detector.detect_defects(image, background_image=bg_image, non_foil_blackout=rem_bg)
    return defect_image[...,::-1], defect_map, timetaken

def video_mod(video: str
              ) -> str:
    """
    Parameters
    ----------
    video : np.ndarray
        Input video as a NumPy array.
    method : str
        Defect detection method to use.
    bg_image : np.ndarray, optional
        Background image as a NumPy array, by default None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the defect video and defect map as NumPy arrays.
    """
    # Your video processing code here...

    return video

# Your Gradio app setup with two tabs for image and video data
methods = ["edge_detector", 
            "background_subtractor",
            "gradient_threshold_detector", 
            "otsu_thresh_detector", 
            "kalman_residual_detector",
            "object_detector"]

with gr.Blocks() as demo:
    gr.Markdown("# Defect Detection Demo")
    gr.Markdown("## Defect detection in foil-like materials using this demo.")
    gr.Markdown("### Prepared by: Ahmad Ali, Github: @ahmadCodes")
    with gr.Tab("Image Data"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="numpy", label="Input Image")
                method_choice = gr.Dropdown(methods, label="Method")
                bg_image_input = gr.Image(type="numpy", label="Background Image (Required if method=background_subtractor)")
                rem_bg_checkbox = gr.Checkbox(label="Preprocessing: Blackout non-foil region")
                
                
            with gr.Column():
                image_output = gr.Image(type="numpy", label="Defect Detection Image")
                defect_map_output = gr.Image(type="numpy", label="Defect Mask Image")
                time_taken_output = gr.Textbox(label="Time taken (seconds)")
                image_button = gr.Button("Process Image")

    with gr.Tab("Video Data"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(type="numpy", label="Input Video")
                video_button = gr.Button("Process Video")
            with gr.Column():   
                video_output = gr.Video(type="numpy", label="Defect Detection Video")


    image_button.click(image_mod,
                       inputs=[image_input, method_choice, bg_image_input, rem_bg_checkbox],
                       outputs=[image_output, defect_map_output, time_taken_output])

    video_button.click(video_mod,
                       inputs=[video_input],
                       outputs=[video_output ])

demo.launch()