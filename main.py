from mtcnn import MTCNN
import cv2

face_detector = MTCNN()
camera = cv2.VideoCapture(0)

def capture_face():
  while True:
    ret, frame = camera.read()


    faces = face_detector.detect_faces(frame)

    for face in faces:
      (x, y, w, h) = face['box']
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Maskya", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  
  camera.release()
  cv2.destroyAllWindows()


if __name__=="__main__":
  capture_face()