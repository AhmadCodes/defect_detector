
import cv2
import numpy as np
import os
#%% useful variables and callbacks initialization
global background
background = None

# for smoothing and convolution fucntions etc
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
kernel = np.ones((5,5),np.uint8)

algo = 'MOG2' # or 'KNN' for KNN background subtraction
nbg_rec = 500 # number of frames to be used for background modeling

#background subtraction builtin methods
if algo == 'MOG2':
    BACK_SUB = cv2.createBackgroundSubtractorMOG2(history = nbg_rec,
                                            varThreshold = 6,
                                            detectShadows = False )
else:
    BACK_SUB = cv2.createBackgroundSubtractorKNN(history = nbg_rec,
                                            dist2Threshold = 100.0,
                                            detectShadows = False )

#%% Helper functions

def movingAvg(image: np.ndarray, beta:float):
    """
    Moving average of the video frames to construct a background model.

    Parameters
    ----------
    image : np.ndarray
        The current frame to be used for background modeling.
    beta : float
        The hyperparameter for moving average.
    """
    
    global background
    # if there is not any background model constructed previously
    if background is None:
        background = image.copy().astype("float")
        return 
    # get the weighted average (method1)
    cv2.accumulateWeighted(image, background, beta)
    # use the builtin method (method2, better)
    BACK_SUB.apply(image) 


def thresholding(image: np.ndarray,
                 background_img: np.ndarray
                 ) -> tuple[np.ndarray,np.ndarray] | tuple[None,None]:
    """
    _summary_

    Parameters
    ----------
    image : np.ndarray
        The current frame to be used to remove the background and detect defects.
    background_img : np.ndarray
        The background image to be used for background subtraction.


    Returns
    -------
    tuple[np.ndarray,np.ndarray] | tuple[None,None]
        A tuple containing the thresholded image and the contours of the objects in the image.\
        If no object is detected, then None is returned.
    """

    #get the delta image between backrgound model and current frame
    delta = cv2.absdiff(background_img.astype("uint8"), image)
    
    # get the thresholded frame
    # otsu thresholding
    thresholded = cv2.threshold(delta, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    col_avg = np.mean(image, axis=0)
    col_avg_normalized = (col_avg - np.min(col_avg)) / ((np.max(col_avg) - np.min(col_avg)) + 1e-6)

    thresholded[:, col_avg_normalized<0.5] = 0
    
    #see if there is any contours in the thresholded frame
    ( cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0: #return nothing if there is no object present
        return None, None
    else: #if there is an object present, return the frame and contours of objects
        return thresholded, cnts 


def drawcontours(image: np.ndarray,
                 contours: any
                 ) -> np.ndarray:
    """
    Draw bounding boxes around the detected defects.

    Parameters
    ----------
    image : np.ndarray
        The current frame to be used to remove the background and detect defects.
    contours : any
        The contours of the objects in the image.

    Returns
    -------
    np.ndarray
        The image with bounding boxes drawn around the detected defects.
    """

    defect_image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    

    # draw recatngles around contours
    for c in contours:
        rect = cv2.boundingRect(c)
        #if the area of contour is within a specific range
        if rect[2] < 5 or rect[3] < 5 or cv2.contourArea(c) > 80000 : continue
        #draw bbox on the image
        defect_image = cv2.rectangle(defect_image, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0,0,255), 2)
        
    return defect_image


#%%

def detect(video: str,
           output_video_dir: str = None,
           debug: bool = False
           ) -> tuple[str,str] :
    """
    Detect defects in a video using background subtraction method.

    Parameters
    ----------
    video : str
        The path to the video to detect defects in.
    output_video_dir : str, optional
        If specified, the output videos will be saved to this dir, by default None
    debug : bool, optional
        Whether to show the processed video, by default False

    Returns
    -------
    tuple[str,str]
        A tuple containing the paths to the annotated defect video and the defect map video.
    """
    cap = cv2.VideoCapture(cv2.samples.findFileOrKeep(video))

    # Initialize useful variables
    n_frame = 0 # For frame counting
    beta = 0.8  # Moving average hyperparameter for background modeling
    
    # Desired output video parameters
    
    
    output_height = 256
    output_width = 3660
    output_fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    
    if output_video_dir is None:
        output_video_dir = "/tmp"
        
    
    # initialize output video writers
    output_vid_path1 = os.path.join(output_video_dir, "bg_sub_defect_detector_defect_vid.avi")
    output_vid_path2 = os.path.join(output_video_dir, "bg_sub_defect_detector_defect_map.avi")
    out_def_vid = cv2.VideoWriter(output_vid_path1,fourcc, output_fps, (output_width,output_height))
    out_def_map_vid = cv2.VideoWriter(output_vid_path2,fourcc, output_fps, (output_width,output_height))
    # loop over the frames of the video
    while(True):
        # get the current frame
        (ret, frame) = cap.read()
        if not isinstance(frame, np.ndarray) or not ret:
            break
        frame = cv2.resize(ret, (output_width, output_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.astype("uint8")

        # smoothen and convert to grayscale
        smoothed = cv2.bilateralFilter(frame,15,75,75)
        gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY) if smoothed.ndim == 3 else smoothed.astype("uint8")
        # gr = cv2.GaussianBlur(gr, (15, 15), 0)
        # gr = cv2.medianBlur(gr,5)
        gray = cv2.bilateralFilter(gray,15,75,75) #smoothen again
        
        # Construct background model
        if n_frame < nbg_rec:     
            movingAvg(gray, beta)
            if n_frame == 1:
                print("Constructing a background model")
            elif n_frame == nbg_rec-1:
                print("background model constructed")
            defect_map = np.zeros_like(frame)
            drawn_image = frame.copy()
            drawn_image = cv2.putText(drawn_image, "Constructing a background model", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        #When background model is constructed for specified number of intitial frames    
        else:
            #get the thresholded foregroound mask
            background_img = BACK_SUB.getBackgroundImage() # backSub is global
            defect_map, contours = thresholding(gray, background_img)
            # If contour/object is detected in the foreground
            if contours is not None:
                #make and show bounding boxes over objects in foreground
                drawn_image = drawcontours(frame,contours)
            else:
                drawn_image = frame.copy()
                
        if debug:
            cv2.imshow("Draw Defects", drawn_image)
            cv2.imshow("Defect Map", defect_map)
            keypress = cv2.waitKey(1) & 0xFF
            if keypress == ord("q"):
                break
        
        # write to output video
        out_def_vid.write(drawn_image)
        out_def_map_vid.write(defect_map)    
        
        n_frame += 1 #update frame number
        # Press Q on keyboard to  exit the loop
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # return the path of the output video
    return output_vid_path1, output_vid_path1


#%% Test the function if run as a script

if __name__ == "__main__":
    
    test_vid_path = "../../data/test_video.avi"
    
    detect(test_vid_path, debug=True)
# %%
