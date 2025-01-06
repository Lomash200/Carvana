import streamlit as st
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp
from keras.models import load_model
import webbrowser
import time

# Initialize session state
if "emotion" not in st.session_state:
    st.session_state["emotion"] = ""
if "frame_count" not in st.session_state:
    st.session_state["frame_count"] = 0
if "run" not in st.session_state:
    st.session_state["run"] = True

# Load model and dependencies
@st.cache_resource
def load_dependencies():
    model = load_model("model.h5")
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    labels = np.load("labels.npy")
    holistic = mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return model, labels, holistic

model, labels, holistic = load_dependencies()

def process_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = holistic.process(frame)
    print("Frames: ", frame, "Results: ", results)
    
    if results.face_landmarks:
        landmarks = []
        for landmark in results.face_landmarks.landmark:
            landmarks.extend([landmark.x - results.face_landmarks.landmark[1].x,
                              landmark.y - results.face_landmarks.landmark[1].y])
        
        # Add placeholder data for hands if not detected
        landmarks.extend([0.0] * 84)  # 21 landmarks * 2 coordinates * 2 hands
        
        landmarks = np.array(landmarks).reshape(1, -1)
        prediction = labels[np.argmax(model.predict(landmarks))]
        
        cv2.putText(frame, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame, prediction
    
    return frame, None

# Streamlit app
st.title("Emotion Based Music Recommender 🎵")
st.write("By Lomash Badole")

# Sidebar for webcam control
st.sidebar.title("Webcam Control")
st.session_state["run"] = st.sidebar.checkbox("Run", value=st.session_state["run"])

# Main content
col1, col2 = st.columns(2)

with col1:
    # Placeholder for webcam feed
    webcam_placeholder = st.empty()

with col2:
    # Emotion display
    emotion_placeholder = st.empty()
    
    # Input fields
    lang = st.text_input("Language", key="lang_input", placeholder="Enter language (e.g. English)")
    singer = st.text_input("Singer", key="singer_input", placeholder="Enter singer name")
    
    if st.button("Recommend me songs 🎵", key="recommend_button"):
        if not st.session_state["emotion"]:
            st.warning("⚠️ Please wait while I capture your emotion first")
        else:
            try:
                search_query = f"https://www.youtube.com/results?search_query={lang}+{st.session_state['emotion']}+song+{singer}"
                webbrowser.open(search_query)
                st.success("🎉 Opening YouTube with your personalized recommendations!")
            except Exception as e:
                st.error(f"Error opening browser: {str(e)}")

# Webcam capture and processing
if st.session_state["run"]:
    webcam = cv2.VideoCapture(0)
    
    # Capture frame
    success, frame = webcam.read()
    if success:
        st.session_state["frame_count"] += 1
        
        # Process frame
        processed_frame, emotion = process_frame(frame)
        
        # Update UI
        webcam_placeholder.image(processed_frame, channels="BGR", use_column_width=True)
        if emotion:
            emotion_placeholder.write(f"Detected Emotion: {emotion}")
            st.session_state["emotion"] = emotion
    else:
        st.error("Failed to capture frame from webcam")
    
    webcam.release()

# Sleep for a short duration to control update frequency
time.sleep(0.1)

# The app will automatically rerun due to Streamlit's execution model