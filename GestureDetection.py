from paho.mqtt import client as mqtt_client
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time


MQTT_CLIENT_ID = "Gesture-Detector"
MQTT_BROKER = ""
MQTT_PORT = -1
USERNAME = ""
PASSWORD = ""
TOPIC = ""

CAP = cv.VideoCapture(0)
CAP.set(cv.CAP_PROP_FRAME_WIDTH, 320)
CAP.set(cv.CAP_PROP_FRAME_HEIGHT, 240)


def analyze_camera_ouput():
    frame_counter = 0
    PROCESS_EVERY_N = CAP.get(cv.CAP_PROP_FPS) // 2

    while True:
        ret, image = CAP.read()
        if not ret:
            break

        frame_counter += 1
        if frame_counter % PROCESS_EVERY_N == 0:
            rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )
            timestamp = int(time.time() * 1000)
            RECOGNIZER.recognize_async(mp_image, timestamp)
            frame_counter = 0

    RECOGNIZER.close()
    CAP.release()


def save_result(result, output_image, timestamp_ms):
    global TOPIC
    global client
    global value

    if not result.gestures:
        value["category"] = "none"
        value["landmark"] = (0, 0, 0)
        client.publish(TOPIC, value["category"])
        return

    category = result.gestures[0][0].category_name
    if not category or category == "none":
        value["category"] = "none"
        value["landmark"] = (0, 0, 0)
        client.publish(TOPIC, value["category"])
        return

    landmark = (
        result.hand_landmarks[0][0].x,
        result.hand_landmarks[0][0].y,
        result.hand_landmarks[0][0].z
    )
    if category != value["category"]:
        value["category"] = category
        value["landmark"] = landmark
        print("Gesture:", category)
        client.publish(TOPIC, value["category"])
        return
        

BASE = python.BaseOptions("gesture_recognizer.task")
OPTIONS = vision.GestureRecognizerOptions(
    base_options=BASE,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=save_result
)
RECOGNIZER = vision.GestureRecognizer.create_from_options(OPTIONS)
value = {
    "category": "none",
    "landmark": (0, 0, 0)
}


def on_connect(client, userdata, flags, reason_code):
    if (reason_code == 0):
        analyze_camera_ouput()
    else:
        print("Not Connected!")
    


def main():
    client = mqtt_client.Client(client_id=MQTT_CLIENT_ID)
    client.username_pw_set(username=USERNAME, password=PASSWORD)
    client.on_connect = on_connect
    client.loop_forever()


if __name__ == "__main__":
    main()
