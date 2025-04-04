import json
import logging
import multiprocessing
import os
import time
import tkinter as tk
import warnings
from datetime import datetime
import cv2
import google.generativeai as genai
import numpy as np
import pyttsx3
import requests
import speech_recognition as sr
import torch
import winsound
from deepface import DeepFace

# torch.hub.set_dir('/tmp/torch/hub')
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)

warnings.simplefilter("ignore", category=FutureWarning)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
gender_net = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")

os.environ['GRPC_VERBOSITY'] = 'ERROR'
logging.getLogger('absl').setLevel(logging.ERROR)

current_time = datetime.now().strftime("%H:%M:%S")
current_date = datetime.now().strftime("%Y-%m-%d")
date_time1 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

recognizer = sr.Recognizer()
recognizer.energy_threshold = 300

gender_classes = ["Male", "Female"]

sleep_mode = True
confidence_threshold = 0.2

engine3 = pyttsx3.init()
desired_voice_id2 = 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0'
current_rate2 = engine3.getProperty('rate')
engine3.setProperty('voice', desired_voice_id2)
engine3.setProperty('rate', current_rate2 - 20)

cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(0)

PHONEME_TO_VISEME = {
    "A": "Open-Wide",
    "E": "Smile",
    "I": "Narrow-Smile",
    "O": "Round",
    "U": "Pucker",
    "M": "Closed",
    "F": "Teeth-Visible"
}

VISEME_COLOR_MAP = {
    "Open-Wide": (255, 0, 0),
    "Smile": (0, 255, 0),
    "Narrow-Smile": (0, 0, 255),
    "Round": (255, 255, 0),
    "Pucker": (255, 0, 255),
    "Closed": (0, 255, 255),
    "Teeth-Visible": (128, 128, 128)
}


def greeting():
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        return "Good morning"
    elif 12 <= current_hour < 18:
        return "Good afternoon"
    else:
        return "Good evening"


def load_api_key(api_key_path):
    with open(api_key_path, 'r') as file:
        api_key = file.read().strip()
    return api_key


def load_syllabus(file_path):
    with open(file_path, 'r') as file:
        syllabus1 = json.load(file)
    return syllabus1


def interpolate_frames(start_color, end_color, progress):
    """Interpolate between two colors based on progress."""
    return tuple(
        int(start_color[i] + (end_color[i] - start_color[i]) * progress)
        for i in range(3)
    )

def generate_phoneme_timings(text, duration_per_char=0.05):
    """Generate phoneme timings based on the input text."""
    phonemes = []
    current_time = 0.0
    for char in text:
        phoneme = {
            "phoneme": char.upper(),
            "start": current_time,
            "end": current_time + duration_per_char
        }
        phonemes.append(phoneme)
        current_time += duration_per_char
    return phonemes

def overlay_lip_animation(frame, phoneme, transition_progress):
    """
    Overlay lip animation based on the current phoneme and transition progress.
    """
    viseme = PHONEME_TO_VISEME.get(phoneme, "Closed")
    base_color = VISEME_COLOR_MAP.get(viseme, (0, 0, 0))

    if transition_progress is not None:
        next_viseme = PHONEME_TO_VISEME.get(phoneme, "Closed")
        next_color = VISEME_COLOR_MAP.get(next_viseme, base_color)
        base_color = interpolate_frames(base_color, next_color, transition_progress)

    frame_height, frame_width, _ = frame.shape
    rect_width = int(frame_width * 0.3)
    rect_height = int(frame_height * 0.2)
    x_start = (frame_width - rect_width) // 2
    y_start = frame_height - rect_height - 50

    # Draw overlay rectangle
    cv2.rectangle(frame, (x_start, y_start), (x_start + rect_width, y_start + rect_height), base_color, -1)

    # Display phoneme and viseme details
    cv2.putText(frame, f"Phoneme: {phoneme}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, base_color, 2)
    cv2.putText(frame, f"Viseme: {viseme}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, base_color, 2)

    return frame

def lip_sync_with_video(video_path, phoneme_timings1, video_state1):
    """
    Synchronize lip movements with the video playback based on phoneme timings.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_time = 1 / frame_rate
    current_time1 = 0.0
    phoneme_index = 0

    while True:
        if video_state1.value == 0:  # Play video
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                current_time1 = 0.0
                phoneme_index = 0
                continue

            # Resize the frame for consistency
            frame = cv2.resize(frame, (1080, 1920), interpolation=cv2.INTER_LINEAR)

            # Handle phoneme timings
            if phoneme_index < len(phoneme_timings1):
                phoneme_data = phoneme_timings1[phoneme_index]
                if phoneme_data["start"] <= current_time1 < phoneme_data["end"]:
                    duration = phoneme_data["end"] - phoneme_data["start"]
                    transition_progress = (current_time1 - phoneme_data["start"]) / duration
                    frame = overlay_lip_animation(frame, phoneme_data["phoneme"], transition_progress)
                elif current_time1 >= phoneme_data["end"]:
                    phoneme_index += 1

            # Display the frame
            cv2.imshow("Sofiya", frame)

            # Exit on 'q' key press
            if cv2.waitKey(int(frame_time * 1000)) & 0xFF == ord('q'):
                break

            current_time1 += frame_time
        else:
            time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

def company_mode(APIKEY, Syllabus, video_state2, stop_switch):
    global dominant_emotion
    AI_name2 = "Sofiya"
    company_name = "MohammedMarwan"
    Api_key = load_api_key(APIKEY)
    genai.configure(api_key=Api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    chat = model.start_chat(history=[])
    engine2 = pyttsx3.init()
    desired_voice_id2 = 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0'
    current_rate2 = engine2.getProperty('rate')
    engine2.setProperty('voice', desired_voice_id2)
    engine2.setProperty('rate', current_rate2 - 20)
    wishing2 = greeting()
    print(f"{wishing2}, my name's {AI_name2}")
    video_state2.value = 0
    engine2.say(f"{wishing2}, my name's {AI_name2}")
    engine2.runAndWait()
    video_state2.value = 1
    timeout_duration = 120
    interests = ["AI", "Robotics", ]
    last_interaction_time = time.time()

    while True:
        ret2, frame2 = cap2.read()
        if ret2:
            result = DeepFace.analyze(frame2, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']
            print(f"Emotion:- {dominant_emotion}")

        elif time.time() - last_interaction_time > timeout_duration:
            print(f"{AI_name2}: No input detected for a long time. Exiting...")
            video_state2.value = 0
            engine2.say("No input detected for a long time. Exiting. Goodbye!")
            engine2.runAndWait()
            video_state2.value = 1
            break

        elif stop_switch.value == 1:  # Check for the stop signal
            print(f"{AI_name2}: Stopping speech. Returning to listening...")
            engine2.stop()
            stop_switch.value = 0  # Reset the stop signal
            continue

        with sr.Microphone() as source:
            print(f"{AI_name2}: Please wait.....")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            winsound.PlaySound("listen.wav", winsound.SND_FILENAME)
            print(f"{AI_name2}: I'm ready.....")
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                last_interaction_time = time.time()  # Reset the timer on successful input
            except sr.WaitTimeoutError:
                continue

        try:
            print("Recognizing speech...")
            question = recognizer.recognize_google(audio)
            print("You said: " + question)

            if question.lower() in ["what is time now"]:
                video_state2.value = 0
                engine2.say(f"The time is {current_time}")
                engine2.runAndWait()
                video_state2.value = 1
                continue

            elif question.lower() in ["talk about latest news"]:
                for topic in interests:
                    print(f"News for {topic}:")
                    news = get_news(topic)
                    for article in news:
                        print(f"- {article}")
                        video_state2.value = 0
                        engine2.say(article)
                        engine2.runAndWait()
                        video_state2.value = 1
                continue

            elif question.lower() in ["exit", "goodbye"]:
                farewell_message1 = f"{AI_name2}: Okay see you take care"
                print(f"{AI_name2}: {farewell_message1}")
                video_state2.value = 0
                engine2.say("Okay see you take care!")
                engine2.runAndWait()
                video_state2.value = 1
                break

            found_answer = False
            for subject, topics in Syllabus.items():
                for topic, questions in topics.items():
                    if question in questions:
                        answer = questions[question]
                        video_state2.value = 1
                        print(f"{AI_name2}: {answer}")
                        engine2.say(answer)
                        engine2.runAndWait()
                        video_state2.value = 0
                        found_answer = True
                        break
                if found_answer:
                    break

            if not found_answer:
                instruction1 = (
                    f"In this chat, respond as you are explaining things based on user emotion {dominant_emotion}."
                    f"Then, answer the question carefully."
                    f"Your name's {AI_name2}. and you were created by {company_name}. Services of {company_name} IoT, Robotics, AI, Android ios application development, web application. Contact no +971-43286466. E-mail sales@infotechuae.com. Location at City Tower 2, opposite the Museum of the Future, Dubai, UAE.")
                # prompt = f"Respond with an appropriate tone based on the detected emotion: {dominant_emotion}. Question: {question}. {instruction1}"
                response = chat.send_message(instruction1 + " " + question)
                response_message = response.text
                clean_txt = response_message.replace("*", "")
                lines = clean_txt.splitlines()
                limited = '\n'.join(lines[:25])

                for line in limited.split('.'):
                    if stop_switch.value == 1:
                        engine2.stop()
                        print(f"{AI_name2}: Speech stopped.")
                        break
                    video_state2.value = 0
                    print(line)
                    engine2.say(line)
                    engine2.runAndWait()
                    video_state2.value = 1

        except sr.UnknownValueError:
            pass
        except sr.WaitTimeoutError:
            print(f"Listening timeout.")

def play_avatar_video(video_path, video_state1, phoneme_timings1):
    """
    Play the avatar video with synchronized lip movements.
    """
    lip_sync_with_video(video_path, phoneme_timings1, video_state1)

def start_tkinter_gui(stop_switch):
    """Creates a Tkinter GUI with a stop switch."""

    def toggle_stop():
        stop_switch.value = 1  # Signal to stop speaking
        print("Stop button pressed!.")

    root = tk.Tk()
    root.title("Stop Switch")
    root.geometry("300x200")

    label = tk.Label(root, text="Assistant Control Panel", font=("Arial", 16))
    label.pack(pady=10)

    stop_button = tk.Button(root, text="Stop Speaking", font=("Arial", 14), bg="red", fg="white", command=toggle_stop)
    stop_button.pack(pady=10)

    check = tk.Label(root, text="Ask anything you like after the beep like sound", font=("Arial", 11))
    check.pack(pady=10)

    exit1 = tk.Label(root, text="Say 'goodbye' to exit", font=("Arial", 11))
    exit1.pack(pady=10)

    root.mainloop()

def get_news(topic):
    url = f"https://newsapi.org/v2/everything?q={topic}&apiKey=b2287811b98a472194da2d223da20a8f"
    response = requests.get(url).json()
    articles = response.get("articles", [])

    news_summary = []
    for article in articles[:3]:  # Get top 3 articles
        news_summary.append(article["title"])

    return news_summary

if __name__ == "__main__":
    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn")
    API_KEY = 'GEMINI_API_KEY'
    SYLLABUS_FILE_PATH = 'syllabus.json'
    AVATAR_VIDEO_PATH = 'Lady2.mp4'
    syllabus = load_syllabus(SYLLABUS_FILE_PATH)

    video_state = multiprocessing.Value('i', 1)
    stop_switch = multiprocessing.Value('i', 0)
    phoneme_timings = generate_phoneme_timings("")

    engine3 = pyttsx3.init()
    desired_voice_id2 = 'HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0'
    current_rate2 = engine3.getProperty('rate')
    engine3.setProperty('voice', desired_voice_id2)
    engine3.setProperty('rate', current_rate2 - 20)

    while cap1.isOpened():
        ret1, frame1 = cap1.read()
        if ret1:
            gender_blob = cv2.dnn.blobFromImage(frame1, 1.0, (227, 227), (78.4623377603, 87.7689143744, 114.895847746),
                                                swapRB=False)
            gender_net.setInput(gender_blob)
            gender_preds = gender_net.forward()
            results = model(frame1)

            for *box, conf, cls in results.xyxy[0]:
                x1, y1, x2, y2 = map(int, box)
                label = f"{model.names[int(cls)]} {conf:.2f}"
                name = f"{model.names[int(cls)]}"
                print(f"Object name :- {name}")

                if name == "person":
                    gender_class_id = np.argmax(gender_preds)
                    gender_label = f"{gender_classes[gender_class_id]}"
                    confidence = gender_preds[0, gender_class_id]
                    print(f"Gender :- {gender_label} {confidence:.2f}")
                    if gender_label == "Male":
                        lab = f"Hello sir how can i help you"
                        engine3.say(lab)
                        engine3.runAndWait()
                    else:
                        lab = f"Hello madam how can i help you"
                        engine3.say(lab)
                        engine3.runAndWait()

                    video_process = multiprocessing.Process(target=play_avatar_video,
                                                            args=(AVATAR_VIDEO_PATH, video_state, phoneme_timings))
                    company_process = multiprocessing.Process(target=company_mode,
                                                              args=(API_KEY, syllabus, video_state, stop_switch))
                    gui_process = multiprocessing.Process(target=start_tkinter_gui, args=(stop_switch,))

                    video_process.start()
                    gui_process.start()
                    company_process.start()

                    company_process.join()

                    video_process.terminate()
                    company_process.terminate()
                    gui_process.terminate()
                else:
                    pass

                cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame1, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # cv2.imshow("Object Detection", frame1)

            if cv2.waitKey(1) == ord("q"):
                break

    cap1.release()
    cv2.destroyAllWindows()
