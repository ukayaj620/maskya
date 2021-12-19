from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import numpy as np
from mtcnn import MTCNN


camera = cv2.VideoCapture(0)
mtcnn_face_detector = MTCNN()
mask_detector = load_model("./models/training_01/classifier_mobile_net_v2.h5")


def preprocess_face(frame, face):
    (x, y, w, h) = face
    extracted_face = frame[y:(y + h), x:(x + w)]
    extracted_face = cv2.resize(extracted_face, (224, 224))
    expanded_face = np.expand_dims(extracted_face, axis=0)
    return preprocess_input(expanded_face)


def capture_face():
    while True:
        ret, frame = camera.read()

        faces = mtcnn_face_detector.detect_faces(frame)

        for face in faces:
            (x, y, w, h) = face['box']
            (mask_confidence, non_mask_confidence) = mask_detector.predict(
                preprocess_face(frame, face['box']))[0]

            if mask_confidence > non_mask_confidence:
                confidence = round(mask_confidence * 100, 2)
                label = "Mask, Confidence: {}%".format(confidence)
                color = (0, 255, 0)
            else:
                confidence = round(non_mask_confidence * 100, 2)
                label = "No Mask, Confidence: {}%".format(confidence)
                color = (0, 0, 255)
            
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.imshow("Maskya", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_face()
