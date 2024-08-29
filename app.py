import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import time

# Function to detect emotion
def detect_emotion(image):
    result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
    return result[0]['emotion']

def main():
    st.set_page_config(page_title="Emotion Detection App", page_icon="😃", layout="wide")
    
    st.title("✨ Emotion Detection App ✨")
    st.write("Detect emotions from faces using our advanced AI!")

    # Create two tabs
    tab1, tab2 = st.tabs(["Live Webcam Detection", "Image Upload Detection"])

    with tab1:
        st.header("📹 Live Webcam Emotion Detection")
        run_live = st.checkbox("Start Live Detection")
        
        col1, col2 = st.columns(2)
        with col1:
            FRAME_WINDOW = st.image([])
        with col2:
            emotion_placeholder = st.empty()

        if run_live:
            cap = cv2.VideoCapture(0)
            last_detection_time = time.time()

            while run_live:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame from webcam")
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame)
                current_time = time.time()
                if current_time - last_detection_time >= 2:  # Detect every 2 seconds
                    try:
                        emotions = detect_emotion(frame)
                        emotion_placeholder.empty()  # Clear previous content
                        emotion_placeholder.subheader("Emotion Probabilities:")
                        
                        # Create a more visually appealing display with progress bars
                        for emotion, probability in emotions.items():
                            emotion_placeholder.write(f"{emotion.capitalize()}: {probability:.2f}%")
                            emotion_placeholder.progress(probability / 100)
                        
                        dominant_emotion = max(emotions, key=emotions.get)
                        emoji_dict = {
                            "angry": "😠", "disgust": "🤢", "fear": "😨",
                            "happy": "😄", "sad": "😢", "surprise": "😲", "neutral": "😐"
                        }
                        emotion_placeholder.markdown(f"### Dominant Emotion:\n{emoji_dict.get(dominant_emotion, '🤔')} {dominant_emotion.capitalize()} ({emotions[dominant_emotion]:.2f}%)")
                        
                        last_detection_time = current_time
                    except Exception as e:
                        emotion_placeholder.error(f"Error in emotion detection: {str(e)}")

            cap.release()

    with tab2:
        st.header("🖼️ Image Upload Emotion Detection")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            img_array = np.array(image)

            if st.button("Detect Emotion"):
                with st.spinner("Analyzing emotions..."):
                    try:
                        emotions = detect_emotion(img_array)
                        
                        st.subheader("Emotion Probabilities:")
                        for emotion, probability in emotions.items():
                            st.progress(probability / 100, f"{emotion.capitalize()}: {probability:.2f}%")
                        
                        # Emoji representation
                        dominant_emotion = max(emotions, key=emotions.get)
                        emoji_dict = {
                            "angry": "😠", "disgust": "🤢", "fear": "😨",
                            "happy": "😄", "sad": "😢", "surprise": "😲", "neutral": "😐"
                        }
                        st.subheader("Dominant Emotion:")
                        st.write(f"{emoji_dict.get(dominant_emotion, '🤔')} {dominant_emotion.capitalize()} ({emotions[dominant_emotion]:.2f}%)")
                    except Exception as e:
                        st.error(f"Error in emotion detection: {str(e)}")

    st.markdown("---")
    st.write("Created with ❤️ using Streamlit and DeepFace")

if __name__ == "__main__":
    main()