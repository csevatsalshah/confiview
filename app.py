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
from pyannote.audio import Pipeline
from pyannote.core import Segment, Timeline, Annotation

# Initialize pyannote.audio pipeline (for speaker diarization)
try:
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=None  # No token needed for free model; adjust if using a private model
    )
except Exception as e:
    st.error(f"Error initializing speaker diarization: {str(e}}")
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

    # Extract full audio
    try:
        audio_file = "temp_audio.wav"
        audio = AudioSegment.from_file(temp_video_path, format="mp4")
        audio.export(audio_file, format="wav")
        st.success("Audio is set for evaluation!")
    except Exception as e:
        st.error(f"Sorry, we couldn’t process your video: {str(e)}")
        st.stop()

    # Perform speaker diarization to separate interviewer and interviewee
    try:
        waveform, sample_rate = librosa.load(audio_file, sr=None)
        diarization = diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate})
        # Convert diarization to timeline of speakers
        speakers = {}
        for segment, track, label in diarization.itertracks(yield_label=True):
            if label not in speakers:
                speakers[label] = []
            speakers[label].append((segment.start, segment.end))

        # Transcribe audio and align with diarization
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            try:
                full_text = recognizer.recognize_google(audio_data)
                st.write("Here’s the full conversation we heard:", full_text)
            except sr.UnknownValueError:
                full_text = "Could not understand audio"
                st.write(full_text)
            except sr.RequestError:
                full_text = "Speech recognition unavailable"
                st.write(full_text)
                st.stop()

        # Identify questions and answers using Gemini (analyze full_text for structure)
        try:
            qa_analysis_prompt = f"""
            Here’s a transcript of an interview conversation:
            {full_text}
            As an experienced interviewer, analyze this transcript to identify distinct question-answer pairs. For each pair, extract:
            - The interviewer’s question.
            - The interviewee’s response.
            Provide the output as a list of tuples, like: [("Question 1", "Answer 1"), ("Question 2", "Answer 2"), ...]. Ensure you separate questions and answers accurately, assuming the interviewer asks questions and the interviewee responds. Handle multiple turns naturally, and be thorough to capture all pairs, even if the structure isn’t perfectly clear.
            """
            qa_response = model.generate_content(qa_analysis_prompt)
            qa_pairs = eval(qa_response.text)  # Assuming Gemini outputs a valid Python list of tuples
        except Exception as e:
            st.error(f"Error identifying questions and answers: {str(e)}")
            qa_pairs = [("Could not determine question", full_text)]  # Fallback
    except Exception as e:
        st.error(f"Error in speaker diarization or transcription: {str(e)}")
        qa_pairs = [("Could not determine question", full_text)]  # Fallback

    # Process each question-answer pair
    for i, (question, answer) in enumerate(qa_pairs, 1):
        with st.expander(f"Question {i} and Answer Analysis"):
            st.write(f"**Interviewer’s Question**: {question}")
            st.write(f"**Your Response**: {answer}")

            with st.spinner(f"Reviewing Question {i}—this may take a moment..."):
                # Verbal analysis (detailed and rich feedback)
                st.write(f"Checking how you answered Question {i} in detail...")
                start_time = time.time()
                evaluation_text = f"""
                Here’s what someone said in an interview:
                Interviewer asked: "{question}"
                They replied: "{answer}"
                As an experienced interviewer, provide a thorough and detailed review of their answer. Focus on clarity, relevance to the question, confidence, structure (how logically they organized their response), enthusiasm, and any specific strengths or weaknesses. Highlight what they did exceptionally well, any areas where they could improve, and offer specific, actionable, and friendly tips for their next interview. Keep it natural, conversational, and supportive, like you’re mentoring them personally.
                """
                try:
                    response = model.generate_content(evaluation_text)
                    verbal_feedback = response.text
                    # Heuristic scoring (capped at 95)
                    verbal_score = 60  # Base score
                    if "clear" in verbal_feedback.lower() or "well-spoken" in verbal_feedback.lower():
                        verbal_score += 15
                    if "confident" in verbal_feedback.lower() or "assured" in verbal_feedback.lower():
                        verbal_score += 10
                    if "well-structured" in verbal_feedback.lower() or "organized" in verbal_feedback.lower():
                        verbal_score += 10
                    if "enthusiastic" in verbal_feedback.lower() or "engaged" in verbal_feedback.lower():
                        verbal_score += 5
                    verbal_score = min(95, verbal_score)
                except Exception as e:
                    st.error(f"We hit a snag reviewing your answer for Question {i}: {str(e)}")
                    verbal_feedback = "Sorry, we couldn’t fully review your answer this time."
                    verbal_score = 50

                # Posture, eye contact, gestures, and body language analysis (detailed)
                st.write(f"Looking closely at your body language for Question {i}...")
                start_time = time.time()
                try:
                    mp_pose = mp.solutions.pose
                    mp_face = mp.solutions.face_detection
                    pose = mp_pose.Pose()
                    face = mp_face.FaceDetection(min_detection_confidence=0.5)
                    cap = cv2.VideoCapture(temp_video_path)
                    posture_score, eye_contact_score, gesture_score, head_tilt_score, frame_count, processed_frames = 0, 0, 0, 0, 0, 0
                    frame_interval = 10  # Sample every 10th frame for better accuracy
                    hand_movement, fidget_count = 0, 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_count += 1
                        if frame_count % frame_interval == 0:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            # Posture (spine, shoulders, head)
                            pose_results = pose.process(frame_rgb)
                            if pose_results.pose_landmarks:
                                left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                                right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                                head_y = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y
                                head_x = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x
                                spine_angle = abs(left_shoulder.y - head_y)
                                shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
                                head_tilt = abs(head_x - (left_shoulder.x + right_shoulder.x) / 2)  # Tilt based on head position
                                if spine_angle < 0.3 and shoulder_diff < 0.1:  # Upright and aligned
                                    posture_score += 1
                                if head_tilt < 0.2:  # Minimal tilt for confidence
                                    head_tilt_score += 1
                                # Hand gestures and fidgeting
                                left_hand = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                                right_hand = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                                hand_distance = abs(left_hand.y - right_hand.y)
                                hand_movement += hand_distance
                                if hand_distance > 0.5:  # Significant movement could indicate fidgeting
                                    fidget_count += 1
                            # Eye contact
                            face_results = face.process(frame_rgb)
                            if face_results.detections:
                                eye_contact_score += 1
                            processed_frames += 1
                    posture_score = min(95, (posture_score / processed_frames) * 100 if processed_frames > 0 else 0)
                    eye_contact_score = min(95, (eye_contact_score / processed_frames) * 100 if processed_frames > 0 else 0)
                    gesture_score = min(95, 50 + ((hand_movement / processed_frames) * 10 - (fidget_count / processed_frames * 5)) if processed_frames > 0 else 50)
                    head_tilt_score = min(95, (head_tilt_score / processed_frames) * 100 if processed_frames > 0 else 0)
                    cap.release()
                    pose.close()
                    face.close()
                except Exception as e:
                    st.error(f"Trouble checking your body language for Question {i}: {str(e)}")
                    posture_score = eye_contact_score = gesture_score = head_tilt_score = 50

                # Tone, speech rate, pauses, and enthusiasm analysis (full audio)
                st.write(f"Listening carefully to your voice for Question {i}...")
                start_time = time.time()
                try:
                    y, sr = librosa.load(audio_file, sr=None)  # Use full audio
                    pauses = len(librosa.effects.split(y))
                    speech_rate = len(answer.split()) / (len(y) / sr)  # Words per second
                    pitch_mean = np.mean(librosa.pitch_tuning(y))
                    pitch_variance = np.var(librosa.pitch_tuning(y))  # Variability for enthusiasm
                    tone_score = min(95, max(0, 100 - (pauses * 3)))  # Fewer penalties for natural pauses
                    speech_rate_score = min(95, 80 - abs(speech_rate - 2.5) * 15)  # Ideal ~2.5 words/sec
                    enthusiasm_score = min(95, tone_score + (pitch_mean * 10) + (pitch_variance * 5))  # Richer enthusiasm
                except Exception as e:
                    st.error(f"Couldn’t assess your voice for Question {i}: {str(e)}")
                    tone_score = speech_rate_score = enthusiasm_score = 50

                # Overall confidence score for this Q&A pair
                confidence_score = min(95, (verbal_score * 0.25) + (posture_score * 0.15) + (eye_contact_score * 0.15) + 
                                       (gesture_score * 0.15) + (tone_score * 0.1) + (speech_rate_score * 0.1) + 
                                       (enthusiasm_score * 0.1) + (head_tilt_score * 0.1))

                # Display results for this pair
                st.write(f"**Feedback on Your Answer**: {verbal_feedback}")
                st.write(f"**Posture**: {posture_score:.1f}/100 - How you held yourself (spine, shoulders)")
                st.write(f"**Eye Contact**: {eye_contact_score:.1f}/100 - Connecting with the interviewer")
                st.write(f"**Hand Gestures**: {gesture_score:.1f}/100 - Adding life to your words")
                st.write(f"**Head Position**: {head_tilt_score:.1f}/100 - Confidence in your posture")
                st.write(f"**Tone**: {tone_score:.1f}/100 - Steadiness and clarity in your voice")
                st.write(f"**Speech Pace**: {speech_rate_score:.1f}/100 - How you timed your words")
                st.write(f"**Enthusiasm**: {enthusiasm_score:.1f}/100 - Energy and engagement in your delivery")
                st.write(f"**Overall Impression for This Answer**: {confidence_score:.1f}/100 - Your impact here")

                # Radar chart for this Q&A pair
                aspects = ["Answer Quality", "Posture", "Eye Contact", "Gestures", "Head Position", "Tone", "Pace", "Enthusiasm"]
                scores = [verbal_score, posture_score, eye_contact_score, gesture_score, head_tilt_score, tone_score, speech_rate_score, enthusiasm_score]
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
                    title=f"Your Strengths for Question {i}",
                    height=500,
                )
                st.plotly_chart(fig)

            st.write("---")  # Separator between Q&A pairs

    # Clean up
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    if os.path.exists(audio_file):
        os.remove(audio_file)
else:
    st.info("Upload your video, and let’s see how you did!")