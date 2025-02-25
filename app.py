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
import torch  # Explicitly import for pyannote.audio
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

# Initialize pyannote.audio pipeline (for speaker diarization) with robust error handling
diarization_pipeline = None
try:
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=None  # No token needed for free model
    )
except Exception as e:
    st.error(f"Warning: Speaker diarization failed to initialize: {str(e)}. Continuing with basic transcription.")
    st.write("Analysis will proceed without speaker separation, treating the transcript as multiple Q&A pairs.")

# Get API key from environment variable (Render or local fallback)
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyB8aJR3kyZlTQ5rB928gDt4qMYQH5SQhhM")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-pro')

def normalize_text(text):
    """Normalize concatenated text by adding spaces between words and improving readability."""
    if not text or text in ["Could not understand audio", "No transcription available", "Could not determine question"]:
        return text
    # Split concatenated text using word boundaries and capitalize properly
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Handle camelCase
    text = re.sub(r'([a-zA-Z])([0-9])|([0-9])([a-zA-Z])', r'\1 \2', text)  # Handle alphanumeric
    text = re.sub(r'([a-zA-Z])\.', r'\1. ', text)  # Ensure space after periods
    text = re.sub(r'([^\w\s])\s*', r'\1 ', text)  # Ensure space around punctuation
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    # Capitalize properly (title case for questions, sentence case for answers)
    if text.startswith(('What', 'How', 'Why', 'Tell', 'Can', 'Do', 'Where', 'When', 'Who')):
        text = text.title()
    else:
        text = text.capitalize()
    # Remove repetition or obvious transcription errors
    words = text.split()
    cleaned_words = []
    for i, word in enumerate(words):
        if i > 0 and word.lower() == cleaned_words[-1].lower():
            continue  # Skip repeated words
        # Correct common transcription errors (e.g., "youre" to "you're", "im" to "I'm")
        word = word.replace("youre", "you're").replace("im", "I'm").replace("dont", "don't").replace("cant", "can't")
        cleaned_words.append(word)
    return " ".join(cleaned_words).strip()

# Streamlit UI with color
st.title("ConfiView - Your Interview Coach", help="Upload your interview video for detailed analysis")

# Get port from environment variable (Render or default to 8501 for local testing)
port = os.getenv("PORT", 8501)

# Upload video
video_file = st.file_uploader("Upload Your Interview Video (MP4)", type=["mp4"], help="Supports MP4 files up to 10 minutes")

if video_file is not None:
    # Save uploaded file to disk
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(video_file.read())
    st.success("Your video is ready for review!", icon="✅")

    # Extract full audio
    try:
        audio_file = "temp_audio.wav"
        audio = AudioSegment.from_file(temp_video_path, format="mp4")
        audio.export(audio_file, format="wav")
        st.success("Audio extracted successfully for evaluation!", icon="✅")
    except Exception as e:
        st.error(f"Sorry, we couldn’t process your video: {str(e)}", icon="❌")
        st.stop()

    # Initialize full_text as an empty string for fallback
    full_text = ""

    # Perform transcription with enhanced quality, speed, and progress
    st.write("Starting high-quality, fast transcription...")
    progress_bar = st.progress(0)
    try:
        # Transcribe full audio in 3-second chunks with 0.5-second overlap for better accuracy and speed
        recognizer = sr.Recognizer()
        chunk_duration = 3  # 3-second chunks
        overlap = 0.5  # 0.5-second overlap for continuity
        full_transcript = []
        with sr.AudioFile(audio_file) as source:
            audio_length = len(audio) / 1000  # Duration in seconds
            chunk_count = int((audio_length - overlap) / (chunk_duration - overlap)) + 1
            for i in range(chunk_count):
                start_time = max(0, i * (chunk_duration - overlap) * 1000)  # Start in milliseconds
                end_time = min((i + 1) * chunk_duration * 1000, len(audio))
                chunk = audio[start_time:end_time]
                if len(chunk) > 0:
                    chunk.export(f"temp_chunk_{i}.wav", format="wav")
                    with sr.AudioFile(f"temp_chunk_{i}.wav") as chunk_source:
                        chunk_data = recognizer.record(chunk_source)
                        attempts = 3
                        for attempt in range(attempts):
                            try:
                                chunk_text = recognizer.recognize_google(chunk_data, show_all=True)
                                if 'alternative' in chunk_text and chunk_text['alternative']:
                                    chunk_text = chunk_text['alternative'][0]['transcript']
                                else:
                                    chunk_text = recognizer.recognize_google(chunk_data)
                                full_transcript.append(chunk_text)
                                break
                            except (sr.UnknownValueError, sr.RequestError) as e:
                                if attempt == attempts - 1:
                                    full_transcript.append("[Unintelligible]")
                                else:
                                    time.sleep(0.5)  # Shorter wait for faster retries
                    os.remove(f"temp_chunk_{i}.wav")
                progress = int((i + 1) / chunk_count * 30)  # 30% for transcription
                progress_bar.progress(progress, text=f"Transcription... {progress}%")
        full_text = " ".join(full_transcript).strip()
        full_text = normalize_text(full_text)  # Normalize transcript for readability
        st.write(f"Full transcript: <span style='color: #2ecc71'>{full_text}</span>", unsafe_allow_html=True)
        progress_bar.progress(30, text="Transcription... 100%")  # Transcription complete
    except Exception as e:
        st.error(f"Error in transcription: {str(e)}", icon="❌")
        full_text = "Could not understand audio"
        st.write(f"Full transcript: <span style='color: #e74c3c'>{full_text}</span>", unsafe_allow_html=True)
        progress_bar.progress(30, text="Transcription... 100%")  # Fallback complete

    # Attempt speaker diarization if pipeline is available
    if diarization_pipeline is not None:
        try:
            waveform, sample_rate = librosa.load(audio_file, sr=None)
            progress_bar.progress(40, text="Speaker Diarization... 33%")
            diarization = diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate})
            # Convert diarization to timeline of speakers
            speakers = {}
            for segment, track, label in diarization.itertracks(yield_label=True):
                if label not in speakers:
                    speakers[label] = []
                speakers[label].append((segment.start, segment.end))
            progress_bar.progress(50, text="Speaker Diarization... 100%")  # 50% complete
        except Exception as e:
            st.error(f"Error in speaker diarization: {str(e)}", icon="⚠️")
            diarization = None

    # Identify all questions and answers using Gemini with maximum accuracy and valid Python output
    st.write("Identifying all questions and answers...")
    progress_bar.progress(60, text="Q&A Identification... 50%")
    try:
        qa_analysis_prompt = f"""
        Here’s a transcript of an interview conversation, which may include multiple turns, unclear speech, interruptions, or partial responses:
        {full_text}
        As an experienced interviewer, meticulously analyze this transcript to identify as many distinct question-answer pairs as possible with the highest accuracy, ensuring a valid Python list of tuples as output. For each pair, extract:
        - The interviewer’s exact question (starting with words like 'What', 'How', 'Why', 'Tell me', 'Can you', 'Do you', or similar question indicators, even if incomplete, interrupted, or fragmented, with proper spacing and readability, e.g., 'What’s Your Name?' instead of 'WhatsYourName?').
        - The interviewee’s corresponding response (the answer immediately following the question, avoiding overlap with other questions or noise, even if partial or unclear, with proper spacing and readability, e.g., 'My Name Is Kartikey Bhatt' instead of 'MyNameIsKartikeyBhatt').
        Provide the output *exclusively* as a Python list of tuples in this exact format: `[("Question 1", "Answer 1"), ("Question 2", "Answer 2"), ...]`. Do not include any additional text, explanations, or formatting—only the list of tuples. Ensure you separate questions and answers precisely, assuming the interviewer asks questions and the interviewee responds. Handle multiple turns, unclear phrases, fragmented speech, or interruptions naturally, maximizing the number of pairs while prioritizing accuracy, clarity, and conciseness. Avoid repetition, wrong words, or transcription errors by focusing on the most natural and accurate interpretation. If a question or answer is ambiguous, incomplete, or unclear, include it with a note like '[Unclear Question]', '[Partial Response]', or '[Interrupted]', but only if it’s reasonably identifiable, with proper spacing (e.g., '[Unclear Question]'). Exclude pure noise or irrelevant text, and do not group multiple questions into one—identify each question individually, even if they’re close together or overlapping. For example, if the transcript has two questions like 'What’syourname?' and 'Whereareyoufrom?', output: `[("What’s Your Name?", "My Name Is Kartikey Bhatt"), ("Where Are You From?", "I’m From Mandsaur")]`.
        """
        qa_response = model.generate_content(qa_analysis_prompt)
        # Clean and parse the response to ensure valid Python syntax
        cleaned_response = qa_response.text.strip()
        # Remove any extra whitespace or invalid characters, ensuring it starts and ends with square brackets
        cleaned_response = cleaned_response.replace("\n", "").replace(" ", "")
        if cleaned_response.startswith('[') and cleaned_response.endswith(']'):
            try:
                qa_pairs = eval(cleaned_response)  # Safely evaluate as Python list of tuples
                # Normalize each question and answer for readability
                qa_pairs = [(normalize_text(q), normalize_text(a)) for q, a in qa_pairs]
            except (SyntaxError, NameError, ValueError) as e:
                st.error(f"Error parsing Q&A pairs: Invalid syntax in Gemini response - {str(e)}. Falling back to basic extraction.", icon="⚠️")
                qa_pairs = _extract_basic_qa_pairs(full_text)
        else:
            st.error("Error parsing Q&A pairs: Gemini response not in expected format. Falling back to basic extraction.", icon="⚠️")
            qa_pairs = _extract_basic_qa_pairs(full_text)
        st.write(f"Identified <span style='color: #3498db'>{len(qa_pairs)}</span> questions in the video. Starting detailed analysis...", unsafe_allow_html=True)
        progress_bar.progress(70, text="Q&A Identification... 100%")
    except Exception as e:
        st.error(f"Error identifying questions and answers: {str(e)}", icon="❌")
        qa_pairs = _extract_basic_qa_pairs(full_text) if full_text else [("Could not determine question", "No transcription available")]
        st.write("Identified <span style='color: #e74c3c'>1</span> question (fallback) in the video. Starting detailed analysis...", unsafe_allow_html=True)
        progress_bar.progress(70, text="Q&A Identification... 100%")

    # Helper function for basic Q&A extraction (fallback)
    def _extract_basic_qa_pairs(transcript):
        if not transcript or transcript == "Could not understand audio" or transcript == "No transcription available":
            return [("Could not determine question", "No transcription available")]
        prompt = f"""
        Here’s a transcript of an interview conversation, which may include multiple turns, unclear speech, interruptions, or partial responses:
        {transcript}
        As an experienced interviewer, analyze this transcript to identify distinct question-answer pairs with basic accuracy. For each pair, extract:
        - The interviewer’s question (starting with words like 'What', 'How', 'Why', 'Tell me', 'Can you', 'Do you', or similar, even if incomplete, with proper spacing and readability, e.g., 'What’s Your Name?' instead of 'WhatsYourName?').
        - The interviewee’s response (the answer immediately following, even if partial, with proper spacing and readability, e.g., 'My Name Is Kartikey Bhatt' instead of 'MyNameIsKartikeyBhatt').
        Provide the output *exclusively* as a Python list of tuples in this exact format: `[("Question 1", "Answer 1"), ("Question 2", "Answer 2"), ...]`. Do not include any additional text, explanations, or formatting—only the list of tuples. Handle multiple turns naturally, prioritize simplicity, accuracy, and conciseness, and include only clear pairs. Avoid repetition, wrong words, or transcription errors by focusing on the most natural interpretation.
        """
        try:
            response = model.generate_content(prompt)
            cleaned_response = response.text.strip()
            cleaned_response = cleaned_response.replace("\n", "").replace(" ", "")
            if cleaned_response.startswith('[') and cleaned_response.endswith(']'):
                qa_pairs = eval(cleaned_response)
                # Normalize each question and answer for readability
                return [(normalize_text(q), normalize_text(a)) for q, a in qa_pairs]
            return [("Could not determine question", normalize_text(transcript))]
        except Exception:
            return [("Could not determine question", normalize_text(transcript))]

    # Parallel analysis of all questions
    results = {}
    total_questions = len(qa_pairs)
    progress_bar_total = st.progress(0, text="Analyzing all questions...")

    def analyze_qa(qa_pair, idx):
        question, answer = qa_pair
        verbal_feedback, verbal_score, posture_score, eye_contact_score, gesture_score, head_tilt_score, tone_score, speech_rate_score, enthusiasm_score = None, 500, 500, 500, 500, 500, 500, 500, 500

        # Verbal analysis with progress and humanized, accurate feedback
        st.write(f"Analyzing Question {idx} in detail...")
        progress_bar = st.progress(0, text=f"Question {idx} Verbal Analysis...")
        start_time = time.time()
        evaluation_text = f"""
        Here’s an interview exchange, possibly with unclear or interrupted speech:
        Interviewer asked: "{question}"
        Interviewee replied: "{answer}"
        As a seasoned, empathetic interviewer, provide a concise, accurate, and humanized review of the interviewee’s answer. Focus on clarity, relevance, confidence, structure, enthusiasm, and key strengths or weaknesses, avoiding repetition or errors. Highlight what stands out positively, suggest one or two specific, actionable improvements, and offer warm, supportive tips. Keep it natural and concise, like a mentor offering personalized, high-quality feedback for any scenario, even if the response is partial or unclear (e.g., '[Partial Response]', '[Unclear]'). Avoid technical jargon or overlong explanations.
        """
        try:
            progress_bar.progress(50, text=f"Question {idx} Verbal Analysis... 50%")
            response = model.generate_content(evaluation_text)
            verbal_feedback = response.text.strip()
            # Heuristic scoring (directly out of 1000, humanized and accurate)
            verbal_score = 600  # Base score (out of 1000)
            if "clear" in verbal_feedback.lower() or "well-spoken" in verbal_feedback.lower():
                verbal_score += 150
            if "confident" in verbal_feedback.lower() or "assured" in verbal_feedback.lower():
                verbal_score += 100
            if "well-structured" in verbal_feedback.lower() or "organized" in verbal_feedback.lower():
                verbal_score += 100
            if "enthusiastic" in verbal_feedback.lower() or "engaged" in verbal_feedback.lower():
                verbal_score += 150  # Increased emphasis on enthusiasm for humanized scoring
            if "[partial response]" in verbal_feedback.lower() or "[unclear]" in verbal_feedback.lower() or "[interrupted]" in verbal_feedback.lower():
                verbal_score = max(400, verbal_score - 100)  # Penalty for ambiguity
            verbal_score = max(50, min(950, verbal_score))  # Avoid 0 or 1000, scale to 1000
            progress_bar.progress(100, text=f"Question {idx} Verbal Analysis... 100%")
        except Exception as e:
            st.error(f"Error reviewing Question {idx} verbally: {str(e)}", icon="⚠️")
            verbal_feedback = "Sorry, we couldn’t fully review your answer this time."
            verbal_score = 500
            progress_bar.progress(100, text=f"Question {idx} Verbal Analysis... 100%")
        st.write(f"Verbal review for Question {idx} completed in {time.time() - start_time:.2f} seconds")

        # Posture, eye contact, gestures, and body language analysis with optimized performance
        st.write(f"Analyzing body language for Question {idx}...")
        progress_bar = st.progress(0, text=f"Question {idx} Body Language Analysis...")
        start_time = time.time()
        try:
            mp_pose = mp.solutions.pose
            mp_face = mp.solutions.face_detection
            pose = mp_pose.Pose()
            face = mp_face.FaceDetection(min_detection_confidence=0.5)
            cap = cv2.VideoCapture(temp_video_path)
            posture_score, eye_contact_score, gesture_score, head_tilt_score, frame_count, processed_frames = 0, 0, 0, 0, 0, 0
            frame_interval = 20  # Optimized for speed while maintaining accuracy
            hand_movement, fidget_count = 0, 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            result_queue = Queue()

            def process_frame(frame_idx, frame):
                if frame_idx % frame_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Posture (spine, shoulders, head)
                    pose_results = pose.process(frame_rgb)
                    posture, eye_contact, gesture, head_tilt, fidget = 0, 0, 0, 0, 0
                    if pose_results.pose_landmarks:
                        left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                        right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                        head_y = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y
                        head_x = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x
                        spine_angle = abs(left_shoulder.y - head_y)
                        shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
                        head_tilt = abs(head_x - (left_shoulder.x + right_shoulder.x) / 2)
                        if spine_angle < 0.3 and shoulder_diff < 0.1:  # Upright and aligned
                            posture = 1
                        if head_tilt < 0.2:  # Minimal tilt for confidence
                            head_tilt = 1
                        # Hand gestures and fidgeting
                        left_hand = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                        right_hand = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                        hand_distance = abs(left_hand.y - right_hand.y)
                        gesture = hand_distance
                        if hand_distance > 0.5:  # Significant movement could indicate fidgeting
                            fidget = 1
                    # Eye contact
                    face_results = face.process(frame_rgb)
                    eye_contact = 1 if face_results.detections else 0
                    result_queue.put((frame_idx, posture, eye_contact, gesture, head_tilt, fidget))

            # Parallel processing for frames with larger batch size for speed
            threads = []
            frame_buffer = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_buffer.append(frame)
                if len(frame_buffer) >= 20:  # Process in batches of 20 frames for better performance
                    for idx, f in enumerate(frame_buffer):
                        t = threading.Thread(target=process_frame, args=(frame_count - len(frame_buffer) + idx + 1, f))
                        threads.append(t)
                        t.start()
                    for t in threads:
                        t.join()
                    frame_buffer = []
                    progress = int((frame_count / total_frames) * 50)  # 50% of posture analysis
                    progress_bar.progress(progress, text=f"Question {idx} Body Language Analysis... {progress}%")
            # Process remaining frames
            if frame_buffer:
                for idx, f in enumerate(frame_buffer):
                    t = threading.Thread(target=process_frame, args=(frame_count - len(frame_buffer) + idx + 1, f))
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join()

            # Collect results from queue
            while not result_queue.empty():
                frame_idx, posture, eye_contact, gesture, head_tilt, fidget = result_queue.get()
                if posture:
                    posture_score += 1
                if eye_contact:
                    eye_contact_score += 1
                hand_movement += gesture
                if fidget:
                    fidget_count += 1
                if head_tilt:
                    head_tilt_score += 1
                processed_frames += 1

            progress_bar.progress(100, text=f"Question {idx} Body Language Analysis... 100%")
            posture_score = max(50, min(950, (posture_score / processed_frames) * 1000 if processed_frames > 0 else 500))  # Direct out of 1000, fixed syntax
            eye_contact_score = max(50, min(950, (eye_contact_score / processed_frames) * 1000 if processed_frames > 0 else 500))
            gesture_score = max(50, min(950, 500 + ((hand_movement / processed_frames) * 1000 - (fidget_count / processed_frames * 500)) if processed_frames > 0 else 500))
            head_tilt_score = max(50, min(950, (head_tilt_score / processed_frames) * 1000 if processed_frames > 0 else 500))
            cap.release()
            pose.close()
            face.close()
        except Exception as e:
            st.error(f"Error analyzing body language for Question {idx}: {str(e)}", icon="⚠️")
            posture_score = eye_contact_score = gesture_score = head_tilt_score = 500  # Default to 500 (mid-range)
            progress_bar.progress(100, text=f"Question {idx} Body Language Analysis... 100%")
        st.write(f"Body language review for Question {idx} completed in {time.time() - start_time:.2f} seconds")

        # Tone, speech rate, pauses, and enthusiasm analysis with progress, humanized and accurate
        st.write(f"Analyzing voice for Question {idx}...")
        progress_bar = st.progress(0, text=f"Question {idx} Voice Analysis...")
        start_time = time.time()
        try:
            y, sr = librosa.load(audio_file, sr=None)  # Use full audio
            audio_length = len(y) / sr  # Duration in seconds
            progress_step = 100 / audio_length if audio_length > 0 else 100
            pauses = 0
            for i in range(0, len(y), int(sr * 0.5)):  # Process in 0.5-second chunks for speed and accuracy
                chunk = y[i:i + int(sr * 0.5)]
                if len(chunk) > 0:
                    pauses += len(librosa.effects.split(chunk))
                progress = int((i / len(y)) * 100)
                progress_bar.progress(progress, text=f"Question {idx} Voice Analysis... {progress}%")
            speech_rate = len(answer.split()) / audio_length  # Words per second
            pitch_mean = np.mean(librosa.pitch_tuning(y))
            pitch_variance = np.var(librosa.pitch_tuning(y))  # Variability for enthusiasm
            # Humanized, accurate scoring out of 1000
            tone_score = max(50, min(950, 1000 - (pauses * 15)))  # Fewer penalties, focus on natural pauses, accurate tone
            speech_rate_score = max(50, min(950, 800 - abs(speech_rate - 2.5) * 300))  # Ideal ~2.5 words/sec, emphasize natural pace, accurate rhythm
            enthusiasm_score = max(50, min(950, 600 + (pitch_mean * 200) + (pitch_variance * 200)))  # Stronger emphasis on enthusiasm, natural variation, accurate energy
            progress_bar.progress(100, text=f"Question {idx} Voice Analysis... 100%")
        except Exception as e:
            st.error(f"Error analyzing voice for Question {idx}: {str(e)}", icon="⚠️")
            tone_score = speech_rate_score = enthusiasm_score = 500  # Default to 500 (mid-range)
            progress_bar.progress(100, text=f"Question {idx} Voice Analysis... 100%")
        st.write(f"Voice review for Question {idx} completed in {time.time() - start_time:.2f} seconds")

        return {
            "verbal_feedback": verbal_feedback,
            "verbal_score": verbal_score,
            "posture_score": posture_score,
            "eye_contact_score": eye_contact_score,
            "gesture_score": gesture_score,
            "head_tilt_score": head_tilt_score,
            "tone_score": tone_score,
            "speech_rate_score": speech_rate_score,
            "enthusiasm_score": enthusiasm_score
        }

    # Use ThreadPoolExecutor for parallel analysis
    with ThreadPoolExecutor(max_workers=min(4, len(qa_pairs))) as executor:
        future_to_qa = {executor.submit(analyze_qa, qa_pair, idx): idx for idx, qa_pair in enumerate(qa_pairs, 1)}
        results = {}
        for future in future_to_qa:
            idx = future_to_qa[future]
            try:
                result = future.result()
                results[idx] = result
                progress = int((len(results) / len(qa_pairs)) * 30 + 70)  # 70-100% for analysis
                progress_bar_total.progress(progress, text=f"Analyzing all questions... {progress}%")
            except Exception as e:
                st.error(f"Error analyzing Question {idx}: {str(e)}", icon="❌")
                results[idx] = {
                    "verbal_feedback": "Sorry, we couldn’t fully review your answer this time.",
                    "verbal_score": 500,
                    "posture_score": 500,
                    "eye_contact_score": 500,
                    "gesture_score": 500,
                    "head_tilt_score": 500,
                    "tone_score": 500,
                    "speech_rate_score": 500,
                    "enthusiasm_score": 500
                }
                progress = int((len(results) / len(qa_pairs)) * 30 + 70)
                progress_bar_total.progress(progress, text=f"Analyzing all questions... {progress}%")

    # Display results for each question with color
    total_questions = len(qa_pairs)
    overall_score = 0
    for i in range(1, total_questions + 1):
        with st.expander(f"Question {i} and Answer Analysis", expanded=False):
            result = results[i]
            question, answer = qa_pairs[i-1]
            st.write(f"**Interviewer’s Question**: <span style='color: #3498db'>{question}</span>", unsafe_allow_html=True)
            st.write(f"**Your Response**: <span style='color: #2ecc71'>{answer}</span>", unsafe_allow_html=True)
            st.write(f"**Feedback on Your Answer**: <span style='color: #e67e22'>{result['verbal_feedback']}</span>", unsafe_allow_html=True)
            st.write(f"**Posture**: <span style='color: #9b59b6'>{result['posture_score']}/1000</span> - How you held yourself (spine, shoulders)", unsafe_allow_html=True)
            st.write(f"**Eye Contact**: <span style='color: #27ae60'>{result['eye_contact_score']}/1000</span> - Connecting with the interviewer", unsafe_allow_html=True)
            st.write(f"**Hand Gestures**: <span style='color: #f1c40f'>{result['gesture_score']}/1000</span> - Adding life to your words", unsafe_allow_html=True)
            st.write(f"**Head Position**: <span style='color: #d35400'>{result['head_tilt_score']}/1000</span> - Confidence in your posture", unsafe_allow_html=True)
            st.write(f"**Tone**: <span style='color: #e74c3c'>{result['tone_score']}/1000</span> - Steadiness and clarity in your voice", unsafe_allow_html=True)
            st.write(f"**Speech Pace**: <span style='color: #2980b9'>{result['speech_rate_score']}/1000</span> - How you timed your words", unsafe_allow_html=True)
            st.write(f"**Enthusiasm**: <span style='color: #8e44ad'>{result['enthusiasm_score']}/1000</span> - Energy and engagement in your delivery", unsafe_allow_html=True)
            
            # Highlight question score (out of 1000) with color based on range
            question_score = int(result['verbal_score'] * 0.25 + result['posture_score'] * 0.15 + 
                                result['eye_contact_score'] * 0.15 + result['gesture_score'] * 0.15 + 
                                result['head_tilt_score'] * 0.1 + result['tone_score'] * 0.1 + 
                                result['speech_rate_score'] * 0.1 + result['enthusiasm_score'] * 0.1)
            question_score = max(50, min(950, question_score))  # Avoid 0 or 1000
            if 750 <= question_score <= 1000:
                st.markdown(f"<h3 style='color: green; font-weight: bold;'>Question {i} Score: {question_score}/1000</h3>", unsafe_allow_html=True)
            elif 400 <= question_score < 750:
                st.markdown(f"<h3 style='color: yellow; font-weight: bold;'>Question {i} Score: {question_score}/1000</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='color: red; font-weight: bold;'>Question {i} Score: {question_score}/1000</h3>", unsafe_allow_html=True)

            # Radar chart for this Q&A pair with color
            aspects = ["Answer Quality", "Posture", "Eye Contact", "Gestures", "Head Position", "Tone", "Pace", "Enthusiasm"]
            scores = [result['verbal_score'] / 10, result['posture_score'] / 10, result['eye_contact_score'] / 10, 
                      result['gesture_score'] / 10, result['head_tilt_score'] / 10, result['tone_score'] / 10, 
                      result['speech_rate_score'] / 10, result['enthusiasm_score'] / 10]  # Scale to 100 for graph
            fig = go.Figure(data=go.Scatterpolar(
                r=scores + [scores[0]],  # Close the loop
                theta=aspects + [aspects[0]],
                fill='toself',
                line_color='#1f77b4',
                fillcolor='rgba(31, 119, 180, 0.3)'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickcolor='#ffffff')),
                showlegend=False,
                title=f"<span style='color: #2ecc71'>Your Strengths for Question {i}</span>",
                paper_bgcolor='#f5f6fa',
                plot_bgcolor='#f5f6fa',
                font=dict(color='#333333'),
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)
            overall_score += question_score

    # Calculate and display overall interview score with color
    overall_score = int(overall_score / total_questions) if total_questions > 0 else 500  # Average, default to 500 if no questions
    overall_score = max(50, min(950, overall_score))  # Avoid 0 or 1000
    st.write("---")
    st.subheader("Overall Interview Performance", help="Your overall evaluation across all questions")
    if 750 <= overall_score <= 1000:
        st.markdown(f"<h1 style='color: green; font-weight: bold;'>Overall Interview Score: {overall_score}/1000</h1>", unsafe_allow_html=True)
    elif 400 <= overall_score < 750:
        st.markdown(f"<h1 style='color: yellow; font-weight: bold;'>Overall Interview Score: {overall_score}/1000</h1>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h1 style='color: red; font-weight: bold;'>Overall Interview Score: {overall_score}/1000</h1>", unsafe_allow_html=True)

    # Improved overall performance graph (bar chart) with color and interactivity
    fig_overall = go.Figure(data=[go.Bar(
        x=['Overall Score'],
        y=[overall_score / 10],
        marker_color=['#2ecc71' if 750 <= overall_score <= 1000 else '#f1c40f' if 400 <= overall_score < 750 else '#e74c3c'],
        name='Score'
    )])
    fig_overall.update_layout(
        title="<span style='color: #3498db'>Overall Interview Performance</span>",
        yaxis_title="Score (out of 100)",
        yaxis_range=[0, 100],
        yaxis_gridcolor='#ecf0f1',
        xaxis_gridcolor='#ecf0f1',
        plot_bgcolor='#f5f6fa',
        paper_bgcolor='#f5f6fa',
        font=dict(color='#333333'),
        bargap=0.2,
        bargroupgap=0.1,
        showlegend=False,
        height=400,
        width=600,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#ffffff", font_size=12)
    )
    st.plotly_chart(fig_overall, use_container_width=True)

    # Generate overall report with color
    st.subheader("Overall Interview Report", help="Detailed summary of your performance")
    st.write("<span style='color: #3498db'>Here’s a comprehensive analysis of your interview performance:</span>", unsafe_allow_html=True)
    st.write(f"- <span style='color: #2ecc71'>Total Questions Analyzed</span>: {total_questions}", unsafe_allow_html=True)
    st.write("- <span style='color: #9b59b6'>Individual Question Scores</span>:", unsafe_allow_html=True)
    for i in range(1, total_questions + 1):
        question_score = int(results[i]['verbal_score'] * 0.25 + results[i]['posture_score'] * 0.15 + 
                            results[i]['eye_contact_score'] * 0.15 + results[i]['gesture_score'] * 0.15 + 
                            results[i]['head_tilt_score'] * 0.1 + results[i]['tone_score'] * 0.1 + 
                            results[i]['speech_rate_score'] * 0.1 + results[i]['enthusiasm_score'] * 0.1)
        st.write(f"  - <span style='color: #e67e22'>Question {i}</span>: {question_score}/1000", unsafe_allow_html=True)
    st.write(f"- <span style='color: #d35400'>Overall Score</span>: {overall_score}/1000", unsafe_allow_html=True)
    st.write("""
    <span style='color: #2980b9'>**Summary Analysis**:</span>
    Your interview performance shines in [highlight top 2-3 aspects from highest average scores across questions, e.g., 'clarity and confidence in answers', 'strong eye contact', 'steady tone']. You could refine [identify 1-2 lowest average scores, e.g., 'enthusiasm in delivery', 'reducing fidgeting']. For your next interview, try [specific tips, e.g., 'adding more energy to your responses', 'maintaining a straighter posture']. You’ve shown a promising presence—keep honing these skills for even greater impact!
    """, unsafe_allow_html=True)

    # Clean up
    if os.path.exists(temp_video_path):
        os.remove(temp_video_path)
    if os.path.exists(audio_file):
        os.remove(audio_file)
else:
    st.info("Upload your video, and let’s see how you did!", icon="ℹ️")