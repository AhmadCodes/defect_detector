# test_image_defect_detector.py
import numpy as np
import cv2
import pytest

import sys
import os
sys.path.insert(0, "../defect_detector")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../data'))

from defect_detector import ImageDefectDetector


# Sample test image 

test_image_path = "data/test_image.jpg"
test_image = cv2.imread(test_image_path)
assert test_image is not None 
print(f"test_image.shape = {test_image.shape}")
background_image_path = "data/background_image.png"
background_image = cv2.imread(background_image_path)
assert background_image is not None
print(f"background_image.shape = {background_image.shape}")

def test_invalid_method_type():
    with pytest.raises(ValueError):
        ImageDefectDetector(method="invalid_method_type")


def test_detect_defects():
    detector = ImageDefectDetector(method="edge_detector")
    defect_image, defect_map, _ = detector.detect_defects(test_image, non_foil_blackout=True)
    # Add assertions to check the results, e.g., using numpy's assert functions


def test_detect_defects_non_foil_blackout():
    detector = ImageDefectDetector(method="otsu_thresh_detector")
    defect_image, defect_map, _ = detector.detect_defects(test_image, non_foil_blackout=False)
    
    # Verify that the detected defects are only within the foil region (assuming non-foil region is blacked out)
    assert np.all(defect_image[test_image == 0] == 0)
    
    
def test_detect_defects_different_methods():
    methods =  ["edge_detector", 
                "background_subtractor",
                "gradient_threshold_detector", 
                "otsu_thresh_detector", 
                "kalman_residual_detector",
                "object_detector"]
    
    for method in methods:
        detector = ImageDefectDetector(method=method)
        if method == "background_subtractor":
            defect_image, defect_map, _ = detector.detect_defects(image=test_image, background_image=background_image, non_foil_blackout=True)
        else:
            defect_image, defect_map, _ = detector.detect_defects(test_image, non_foil_blackout=True)
        
        # If no error, then the test passes
        assert True


def test_non_foil_blackout_enabled():
    detector = ImageDefectDetector()
    defect_image_with_blackout, _, _ = detector.detect_defects(test_image, non_foil_blackout=True)

    # Verify that the detected defects are only within the foil region (assuming non-foil region is blacked out)
    assert np.all(defect_image_with_blackout[test_image == 0] == 0)

def test_non_foil_blackout_disabled():
    detector = ImageDefectDetector()
    defect_image_without_blackout, _, _ = detector.detect_defects(test_image, non_foil_blackout=False)

    # if no error, then the test passes
    assert True


def test_invalid_inputs():
    detector = ImageDefectDetector()

    with pytest.raises(ValueError):
        # Pass an invalid image (e.g., None)
        detector.detect_defects(None, non_foil_blackout=True)
        
        
    
    #test invalid method type
    with pytest.raises(ValueError):
        detector = ImageDefectDetector(method="invalid_method_type")



def test_performance():
    detector = ImageDefectDetector()
    
    # Measure the time taken for defect detection
    import time
    start_time = time.time()
    defect_image, defect_map, _ = detector.detect_defects(test_image, non_foil_blackout=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    # Assert that the time taken is within acceptable limits
    assert elapsed_time < 5.0  
