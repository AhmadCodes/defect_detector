import gradio as gr
import os
import numpy as np
from defect_detector import ImageDefectDetector, VideoDefectDetector


def image_mod(
    image: np.ndarray,
    method: str,
    bg_image: np.ndarray | None = None,
    rem_bg: bool = False,
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
    print(
        f"Inputs: \n\
        image.shape = {image.shape}, \n\
        method = {method}, \n\
        bg_image.shape = {bg_image.shape if bg_image is not None else None}, \n\
        rem_bg = {rem_bg}"
    )
    defect_detector = ImageDefectDetector(method=method)
    defect_image, defect_map, _, timetaken = defect_detector.detect_defects(
        image, background_image=bg_image, non_foil_blackout=rem_bg
    )
    return defect_image[..., ::-1], defect_map, timetaken


def video_mod(video: str, debug: bool = False) -> str:
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

    defect_detector = VideoDefectDetector(method="background_subtractor")

    output_defect_video, output_mask_video, _ = defect_detector.detect_defects(
        video, debug=debug
    )

    return output_defect_video, output_mask_video


# Your Gradio app setup with two tabs for image and video data
methods = [
    "edge_detector",
    "background_subtractor",
    "gradient_threshold_detector",
    "otsu_thresh_detector",
    "kalman_residual_detector",
    "object_detector",
]

example_test_image_path = os.path.join(
    os.path.dirname(__file__), "data", "test_image.jpg"
)
example_test_bg_image_path = os.path.join(
    os.path.dirname(__file__), "data", "background_image.png"
)
example_test_video_path = os.path.join(
    os.path.dirname(__file__), "data", "test_video.mp4"
)


with gr.Blocks() as demo:
    gr.Markdown("# Defect Detection Demo")
    gr.Markdown(
        "## Defect detection in foil-like materials moving along conveyor belts with top view camera."
    )
    gr.Markdown("### Prepared by: Ahmad Ali, Github: @ahmadCodes")
    with gr.Tab("Image Data"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="numpy", label="Input Image")
                method_choice = gr.Dropdown(methods, label="Method")
                bg_image_input = gr.Image(
                    type="numpy",
                    label="Background Image (Required if method=background_subtractor)",
                )
                rem_bg_checkbox = gr.Checkbox(
                    label="Preprocessing: Blackout non-foil region"
                )

            with gr.Column():
                image_output = gr.Image(type="numpy", label="Defect Detection Image")
                defect_map_output = gr.Image(type="numpy", label="Defect Mask Image")
                time_taken_output = gr.Textbox(label="Time taken (seconds)")
                image_button = gr.Button("Process Image")
        with gr.Row():
            gr.Examples([example_test_image_path], image_input, label="Example Image")
            gr.Examples(
                [example_test_bg_image_path],
                bg_image_input,
                label="Example Background Image",
            )

    with gr.Tab("Video Data"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(type="file", label="Input Video")
                video_button = gr.Button("Process Video")
                debug = gr.Checkbox(
                    label="Debug (Will show processed video via OpenCV HighGUI)"
                )
            with gr.Column():
                video_output = gr.Video(type="file", label="Defect Detection Video")
                video_output2 = gr.Video(type="file", label="Defect Mask Video")
        gr.Examples([example_test_video_path], video_input)

    image_button.click(
        image_mod,
        inputs=[image_input, method_choice, bg_image_input, rem_bg_checkbox],
        outputs=[image_output, defect_map_output, time_taken_output],
    )

    video_button.click(
        video_mod, inputs=[video_input, debug], outputs=[video_output, video_output2]
    )

demo.launch()
