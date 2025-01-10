import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Default demo image
DEMO_IMAGE = 'stand.jpg'

# Define the body parts and pose pairs
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

# Network input dimensions
width = 368
height = 368
inWidth = width
inHeight = height

# Load the pre-trained model
net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

# Streamlit app title
st.title("Human Pose Estimation with OpenCV")

# Instructions
st.text('Upload a clear image with all body parts visible for best results.')

# File uploader for user to upload an image
img_file_buffer = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if img_file_buffer is not None:
    # Read uploaded image
    image = np.array(Image.open(img_file_buffer))
else:
    # Use demo image if no file is uploaded
    demo_image = DEMO_IMAGE
    image = np.array(Image.open(demo_image))

# Display the original image
st.subheader('Original Image')
st.image(image, caption=f"Original Image", use_column_width=True)

# Threshold slider for detecting key points
thres = st.slider('Threshold for detecting key points', min_value=0, value=20, max_value=100, step=5)
thres = thres / 100  # Normalize threshold to 0-1 range

# Pose detection function
@st.cache
def poseDetector(frame):
    # Ensure the input frame has 3 channels
    if len(frame.shape) == 3 and frame.shape[2] == 4:  # RGBA to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    elif len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):  # Grayscale to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    # Prepare the input blob for the network
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    
    # Perform forward pass
    out = net.forward()
    out = out[:, :19, :, :]  # Ensure correct slicing of output tensor
    
    assert len(BODY_PARTS) == out.shape[1]
    
    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thres else None)
        
    # Draw skeleton
    for pair in POSE_PAIRS:
        partFrom, partTo = pair
        assert partFrom in BODY_PARTS
        assert partTo in BODY_PARTS

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
    
    # Measure performance
    t, _ = net.getPerfProfile()
    
    return frame

# Run the pose detector and display the output
output = poseDetector(image)

st.subheader('Estimated Poses')
st.image(output, caption=f"Estimated Poses", use_column_width=True)

st.markdown('''
            # 
            This is a simple human pose estimation app built using OpenCV and Streamlit.
            ''')




