import streamlit as st
import cv2
import google.generativeai as genai
import speech_recognition as sr
import librosa
import numpy as np
import plotly.graph_objects as go
from pydub import AudioSegment
import mediapipe as mp
import os
import subprocess
import time
import re

# Verify ffmpeg is available in PATH
try:
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
    st.write(f"Using video processing tool version: {result.stdout.splitlines()[0]}")
except subprocess.CalledProcessError as e:
    st.error(f"Video processing tool failed: {e.stderr}")
    st.stop()
except FileNotFoundError:
    st.error("Video processing tool not found. Please ensure it’s installed and accessible.")
    st.stop()

# Get API key from Streamlit secrets with fallback for local testing
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.error("API key not found in secrets. Please set 'GEMINI_API_KEY' in Streamlit Cloud or use a hardcoded key for local testing.")
    st.write("For local testing, uncomment the line below and replace with your key:")
    # API_KEY = "YOUR_API_KEY_HERE"  # Hardcode for local testing only
    st.stop()

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-pro')

# Streamlit UI
st.title("ConfiView - Your Interview Coach")

# Upload video
video_file = st.file_uploader("Upload Your Interview Video (MP4)", type=["mp4"])

if video_file is not None:
    # Save uploaded file to disk
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(video_file.read())
    st.success("Your video is ready for review!")

    # Extract audio (limit to first 10 seconds for performance)
    try:
        audio_file = "temp_audio.wav"
        audio = AudioSegment.from_file(temp_video_path, format="mp4")
        audio = audio[:10 * 1000]  # First 10 seconds
        audio.export(audio_file, format="wav")
        st.success("Audio is set for evaluation!")
    except Exception as e:
        st.error(f"Sorry, we couldn’t process your video: {str(e)}")
        st.stop()

    # Audio to text
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            st.write("Here’s what we heard:", text)
        except sr.UnknownValueError:
            text = "Could not understand audio"
            st.write(text)
        except sr.RequestError:
            text = "Speech recognition unavailable"
            st.write(text)
            st.stop()

    # Input question and trigger analysis
    question = st.text_input("What was the interviewer’s question?")
    analyze_button = st.button("Review My Performance")

    if analyze_button and question:
        with st.spinner("Taking a closer look at your performance..."):
            # Filter out question from answer
            answer = re.sub(re.escape(question), "", text, flags=re.IGNORECASE).strip()
            if not answer:
                answer = text  # Fallback if question isn’t in answer

            # Verbal analysis
            st.write("Checking how you answered...")
            start_time = time.time()
            evaluation_text = f"""
            Here’s what someone said in an interview:
            Interviewer asked: "{question}"
            They replied: "{answer}"
            As an experienced interviewer, give them feedback on their answer. Look at how clear and relevant it is, how confident they sound, whether it’s well-structured, and if they seem enthusiastic. Tell them what they did well, what could be better, and share some friendly tips for improvement. Keep it natural, like you’re chatting with them.
            """
            try:
                response = model.generate_content(evaluation_text)
                verbal_feedback = response.text
                # Heuristic scoring (capped at 95)
                verbal_score = 60  # Base score
                if "clear" in verbal_feedback.lower():
                    verbal_score += 15
                if "confident" in verbal_feedback.lower():
                    verbal_score += 10
                if "well-structured" in verbal_feedback.lower() or "organized" in verbal_feedback.lower():
                    verbal_score += 10
                verbal_score = min(95, verbal_score)
            except Exception as e:
                st.error(f"We hit a snag reviewing your answer: {str(e)}")
                verbal_feedback = "Sorry, we couldn’t fully review your answer this time."
                verbal_score = 50
            st.write(f"Answer review completed in {time.time() - start_time:.2f} seconds")

            # Posture, eye contact, and gestures analysis (optimized)
            st.write("Looking at your body language...")
            start_time = time.time()
            try:
                mp_pose = mp.solutions.pose
                mp_face = mp.solutions.face_detection
                pose = mp_pose.Pose()
                face = mp_face.FaceDetection(min_detection_confidence=0.5)
                cap = cv2.VideoCapture(temp_video_path)
                posture_score, eye_contact_score, gesture_score, frame_count, processed_frames = 0, 0, 0, 0, 0
                frame_interval = 30  # Sample every 30th frame for better performance
                hand_movement = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    if frame_count % frame_interval == 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Posture (spine and shoulders)
                        pose_results = pose.process(frame_rgb)
                        if pose_results.pose_landmarks:
                            left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                            right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                            head_y = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y
                            spine_angle = abs(left_shoulder.y - head_y)
                            shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
                            if spine_angle < 0.3 and shoulder_diff < 0.1:
                                posture_score += 1
                            # Hand gestures
                            left_hand = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
                            right_hand = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
                            hand_movement += abs(left_hand - right_hand)
                        # Eye contact
                        face_results = face.process(frame_rgb)
                        if face_results.detections:
                            eye_contact_score += 1
                        processed_frames += 1
                posture_score = min(95, (posture_score / processed_frames) * 100 if processed_frames > 0 else 0)
                eye_contact_score = min(95, (eye_contact_score / processed_frames) * 100 if processed_frames > 0 else 0)
                gesture_score = min(95, 50 + (hand_movement / processed_frames * 10) if processed_frames > 0 else 50)
                cap.release()
                pose.close()
                face.close()
            except Exception as e:
                st.error(f"Trouble checking your body language: {str(e)}")
                posture_score = eye_contact_score = gesture_score = 50
            st.write(f"Body language review completed in {time.time() - start_time:.2f} seconds")

            # Tone, speech rate, and enthusiasm analysis
            st.write("Listening to your voice...")
            start_time = time.time()
            try:
                y, sr = librosa.load(audio_file, duration=10)  # First 10 seconds for speed
                pauses = len(librosa.effects.split(y))
                speech_rate = len(answer.split()) / (len(y) / sr)  # Words per second
                pitch_mean = np.mean(librosa.pitch_tuning(y))
                tone_score = min(95, max(0, 100 - (pauses * 5)))  # Steady tone
                speech_rate_score = min(95, 80 - abs(speech_rate - 2.5) * 20)  # Ideal ~2.5 words/sec
                enthusiasm_score = min(95, tone_score + (pitch_mean * 10))  # Pitch boosts enthusiasm
            except Exception as e:
                st.error(f"Couldn’t assess your voice: {str(e)}")
                tone_score = speech_rate_score = enthusiasm_score = 50
            st.write(f"Voice review completed in {time.time() - start_time:.2f} seconds")

            # Overall confidence score
            confidence_score = min(95, (verbal_score * 0.25) + (posture_score * 0.15) + (eye_contact_score * 0.15) + 
                                   (gesture_score * 0.15) + (tone_score * 0.1) + (speech_rate_score * 0.1) + 
                                   (enthusiasm_score * 0.1))

            # Display results
            st.subheader("Your Performance Review")
            st.write(f"**Question Asked**: {question}")
            st.write(f"**Your Response**: {answer}")
            st.write(f"**Feedback on Your Answer**: {verbal_feedback}")
            st.write(f"**Posture**: {posture_score:.1f}/100 - How you held yourself")
            st.write(f"**Eye Contact**: {eye_contact_score:.1f}/100 - Connecting with the interviewer")
            st.write(f"**Hand Gestures**: {gesture_score:.1f}/100 - Adding life to your words")
            st.write(f"**Tone**: {tone_score:.1f}/100 - Steadiness in your voice")
            st.write(f"**Speech Pace**: {speech_rate_score:.1f}/100 - How you timed your words")
            st.write(f"**Enthusiasm**: {enthusiasm_score:.1f}/100 - Energy in your delivery")
            st.write(f"**Overall Impression**: {confidence_score:.1f}/100 - Your total vibe")

            # Radar chart
            aspects = ["Answer Quality", "Posture", "Eye Contact", "Gestures", "Tone", "Pace", "Enthusiasm"]
            scores = [verbal_score, posture_score, eye_contact_score, gesture_score, tone_score, speech_rate_score, enthusiasm_score]
            fig = go.Figure(data=go.Scatterpolar(
                r=scores + [scores[0]],  # Close the loop
                theta=aspects + [aspects[0]],
                fill='toself',
                line_color='#1f77b4',
                fillcolor='rgba(31, 119, 180, 0.3)'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                title="Your Interview Strengths",
                height=500,
            )
            st.plotly_chart(fig)

        # Clean up
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(audio_file):
            os.remove(audio_file)
else:
    st.info("Upload your video, and let’s see how you did!")