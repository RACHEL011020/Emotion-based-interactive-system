from time import sleep
import cv2
import numpy as np
import tensorflow as tf
from statistics import mode

from model import predict, image_to_tensor, deepnn
from tts import convert_CSV_to_dict, drama_manager, play_utterance, AUDIOSAVE

# Load the Haar Cascade for face detection
CASC_PATH = './data/haarcascade_files/haarcascade_frontalface_default.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)

# List of emotions
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

SCRIPT_CSV = "test_workshop3-2.csv"

def print_instructions():
    print("\nPress:")
    print("Enter scene number")
    print("Space to stop analysis")
    print("Q to quit")


def most_common(List):
    return mode(List)


def format_image(image):
    """
    Detects multiple faces and formats them for prediction.
    :param image: Input image frame.
    :return: A list of (face_image, face_coordinates) for each detected face.
    """
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade_classifier.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) == 0:
        return []
    
    formatted_faces = []
    for (x, y, w, h) in faces:
        face = image[y:y + h, x:x + w]
        try:
            face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_CUBIC)
            formatted_faces.append((face, (x, y, w, h)))
        except Exception as e:
            print(f"[+] Problem during resize: {e}")
            continue

    return formatted_faces


def demo(modelPath, showBox=False):
    """
    Main function to capture video, detect faces, and predict emotions for multiple faces.
    :param modelPath: Path to the pre-trained model.
    :param showBox: Whether to show bounding boxes around detected faces.
    """
    face_x = tf.compat.v1.placeholder(tf.float32, [None, 2304])
    y_conv = deepnn(face_x)
    probs = tf.nn.softmax(y_conv)

    saver = tf.compat.v1.train.Saver()
    ckpt = tf.train.get_checkpoint_state(modelPath)
    sess = tf.compat.v1.Session()
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    feelings_faces = []
    for emotion in EMOTIONS:
        feelings_faces.append(cv2.imread(f'./data/emojis/{emotion}.png', -1))

    video_captor = cv2.VideoCapture(0)
    convert_CSV_to_dict(SCRIPT_CSV)

    while True:
        ret, frame = video_captor.read()
        if not ret:
            break

        detected_faces = format_image(frame)  # Detect and format all faces

        if showBox and detected_faces:
            for _, (x, y, w, h) in detected_faces:
                cv2.rectangle(img=frame, pt1=(x, y), pt2=(x+w, y+h), color=(255, 0, 0), thickness=2)

        for face_image, (x, y, w, h) in detected_faces:
            tensor = image_to_tensor(face_image)
            result = sess.run(probs, feed_dict={face_x: tensor})

            if result is not None:
                emotion_index = np.argmax(result[0])
                emotion = EMOTIONS[emotion_index]

                # Display the emotion label next to the face
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display emoji corresponding to detected emotion
                emoji_face = feelings_faces[emotion_index]
                cv2.imshow('Emoji Display', emoji_face)

        # Display the frame with face rectangles and emotion labels
        cv2.imshow('face', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Space to stop analysis (additional logic can go here)
            print("Stopped analysis.")
            break

        sleep(0.04)

    video_captor.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    demo(modelPath='./model_checkpoint')
