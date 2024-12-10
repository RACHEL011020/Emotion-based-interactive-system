# /usr/bin/python3
from time import sleep
import cv2
import numpy as np
import sys
import tensorflow as tf
import statistics
from statistics import mode

from model import predict, image_to_tensor, deepnn
from tts import convert_CSV_to_dict, drama_manager, play_utterance, AUDIOSAVE


CASC_PATH = './data/haarcascade_files/haarcascade_frontalface_default.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
EMOTIONS = ['angry', 'disgusted', 'fearful',
            'happy', 'sad', 'surprised', 'neutral']

SCRIPT_CSV = "test_workshop3-2.csv"

def print_instructions():
    print("\nPress:")
    # print("1 angry, 2 disgusted, 3 fearful, 4 happy, 5 sad, 6 surprised, 7 neutral")
    print("Enter scene number")
    print("Space to stop analysis")
    print("Q to quit")


def most_common(List):
    return (mode(List))


def format_image(image):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=5
    )
    # None is no face found in image
    if len(faces) <= 0:
        return None, None
    max_are_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
            max_are_face = face
    # face to image
    face_coor = max_are_face
    image = image[face_coor[1]:(
        face_coor[1] + face_coor[2]), face_coor[0]:(face_coor[0] + face_coor[3])]
    # Resize image to network size
    try:
        image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC)
    except Exception:
        print("[+} Problem during resize")
        return None, None
    return image, face_coor


def face_dect(image):
    """
    Detecting faces in image
    :param image:
    :return:  the coordinate of max face
    """
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor=1.3,
        minNeighbors=5
    )
    if len(faces) <= 0:
        return None
    max_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_face[2] * max_face[3]:
            max_face = face
    face_image = image[max_face[1]:(
        max_face[1] + max_face[2]), max_face[0]:(max_face[0] + max_face[3])]
    try:
        image = cv2.resize(face_image, (48, 48),
                           interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("[+} Problem during resize")
        return None
    return face_image


def resize_image(image, size):
    try:
        image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC) / 255.
    except Exception:
        print("+} Problem during resize")
        return None
    return image


def draw_emotion():
    pass


def demo(modelPath, showBox=False):
    face_x = tf.compat.v1.placeholder(tf.float32, [None, 2304])
    y_conv = deepnn(face_x)
    probs = tf.nn.softmax(y_conv)

    saver = tf.compat.v1.train.Saver()
    ckpt = tf.train.get_checkpoint_state(modelPath)
    sess = tf.compat.v1.Session()
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        # print('Restore model sucsses!!\nNOTE: Press SPACE on keyboard to capture face.')

    feelings_faces = []
    for index, emotion in enumerate(EMOTIONS):
        feelings_faces.append(cv2.imread(f'./data/emojis/{emotion}.png', -1))

    video_captor = cv2.VideoCapture(0)
    # video_captor = cv2.VideoCapture(2)

    emoji_face = []
    result = None
    # print_instructions()
    convert_CSV_to_dict(SCRIPT_CSV)
    while True:
        ret, frame = video_captor.read()
        detected_face, face_coor = format_image(frame)
        if showBox and face_coor is not None:
            [x, y, w, h] = face_coor
            cv2.rectangle(img=frame, pt1=(x, y), pt2=(x+w, y+h), color=(255, 0, 0), thickness=2)

        emoji_face = []
        result = None
        frame_count = 0
        wanted_emotion_count = []
        counting_frames = False
        wanted_emotion = None

        print_instructions()
        while True:
            ret, frame = video_captor.read()
            # print(f'frame: {frame}')
            frame_1 = np.zeros((512, 512, 1), dtype="uint8")
            frame_3 = np.zeros((512, 512, 3), dtype="uint8")
            detected_face, face_coor = format_image(frame)
            if showBox and face_coor is not None:
                [x, y, w, h] = face_coor
                # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.rectangle(img=frame, pt1=(x, y), pt2=(x+w, y+h), color=(255, 0, 0), thickness=2)

        # if cv2.waitKey(1) & 0xFF == ord(' '):

        #   if detected_face is not None:
        #     cv2.imwrite('a.jpg', detected_face)
        #     tensor = image_to_tensor(detected_face)
        #     result = sess.run(probs, feed_dict={face_x: tensor})
        #     # print(result)

            sleep(0.04)
            if detected_face is not None:
                cv2.imwrite('a.jpg', detected_face)
                tensor = image_to_tensor(detected_face)
                result = sess.run(probs, feed_dict={face_x: tensor})
                # print(result)
            if result is not None:
                for index, emotion in enumerate(EMOTIONS):
                    cv2.namedWindow(winname="Emoji Display", flags=cv2.WINDOW_AUTOSIZE)
                    cv2.putText(img=frame, text=emotion, org=(10, index * 20 + 20), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.5, color=(0, 255, 0), thickness=1)
                    cv2.rectangle(img=frame, pt1=(130, index * 20 + 10), pt2=(130 + int(result[0][index] * 100), (index + 1) * 20 + 4), color=(255, 0, 0), thickness=-1)
                    emoji_face = feelings_faces[np.argmax(result[0])]
                    # cv2.putText(img=emoji_face, text=emotion, org=(10, index * 20 + 20), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.5, color=(0, 255, 0), thickness=1)
                    # cv2.rectangle(img=emoji_face, pt1=(200, 300), pt2=(500, 300), color=(255, 0, 0), thickness=-1)
                    cv2.imshow(winname='Emoji Display', mat=emoji_face)
                    if counting_frames is True:
                        wanted_emotion_count.append(np.argmax(result[0]))

                # for c in range(3):
                #     frame[200:320, 10:130, c] = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0) + frame[200:320, 10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)
            cv2.imshow('face', frame)
            detected_key = cv2.waitKey(1) & 0xFF
            if wanted_emotion is None:
                if detected_key == ord('1'):
                    wanted_emotion = 2
                    scene = 1
                elif detected_key == ord('2'):
                    wanted_emotion = 3
                    scene = 2
                elif detected_key == ord('3'):
                    wanted_emotion = 3
                    scene = 3
                elif detected_key == ord('4'):
                    wanted_emotion = 0
                    scene = 4
                elif detected_key == ord('5'):
                    wanted_emotion = 4
                    scene = 5
                elif detected_key == ord('6'):
                    wanted_emotion = 4
                    scene = 6
                elif detected_key == ord('7'):
                    wanted_emotion = 3
                    scene = 7
                if wanted_emotion is not None:
                    print(f"scene: {scene}, wanted emotion: { EMOTIONS[wanted_emotion] }")
                    counting_frames = True
            if detected_key == ord(' ') and wanted_emotion is not None:
                # if space is pressed, program stops analysing specified emotion
                print(f"{wanted_emotion == most_common(wanted_emotion_count)}")
                print(wanted_emotion_count)
                if (most_common(wanted_emotion_count) == wanted_emotion):
                    play_utterance(f"{AUDIOSAVE}scene-{scene}-yes.mp3")
                else:
                    play_utterance(f"{AUDIOSAVE}scene-{scene}-no.mp3")
                counting_frames = False
                wanted_emotion = None
                scene = None
                print_instructions()
                wanted_emotion_count = []

            if detected_key == ord('q'):
                break
