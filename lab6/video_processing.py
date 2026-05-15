import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

VIDEO_DIR = ".\\data"
MODEL_FILE = ".\\model.h5"

img_height = 180
img_width = 180

model = load_model(MODEL_FILE)
class_names = ['cats', 'dogs']

def analyze_video(filepath):
    cap = cv2.VideoCapture(filepath)
    frame_count = 0
    frames_to_skip = 80
    saved_res = ['',0]
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (800, 500))
        if frame_count % frames_to_skip == 0:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized_frame = cv2.resize(rgb_frame, (img_width, img_height))

                img_array = tf.keras.utils.img_to_array(resized_frame)
                img_array = tf.expand_dims(img_array, 0)

                predictions = model.predict(img_array, verbose=0)
                score = tf.nn.softmax(predictions[0])

                class_idx = np.argmax(score)
                label = class_names[class_idx]
                confidence = 100 * np.max(score)
                if label == "dogs":
                    label = "dog"
                else:
                    label = "cat"

                saved_res[0] = label
                saved_res[1] = confidence
                print(f'{label}: {confidence:.2f}')
                print()

            except Exception as e:
                print(f"Error: {e}")
                continue

        cv2.putText(frame,
                    f'{saved_res[0]} ({saved_res[1]:.2f}%)',
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,255,0),
                    2)
        cv2.imshow("Video", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def show_video(filepath):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (800, 500))
        cv2.imshow(f"Video {filepath}", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if '__main__' == __name__:
    mp4_list = ['2_dogs.mp4', '3_dogs.mp4', '3_cats.mp4', 'cats_3.mp4']
    mp4_path = []
    for name in mp4_list:
        mp4_path.append(os.path.join(VIDEO_DIR, name))
    #for filename in mp4_path:
    #    show_video(filename)

    print('Перевірка роботи навченої моделі на відео')
    print(model.summary())
    for filename in mp4_path:
        analyze_video(filename)
