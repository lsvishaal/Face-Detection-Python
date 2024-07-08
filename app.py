import cv2
from mtcnn import MTCNN

# Initialize the MTCNN face detector
detector = MTCNN()

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame
    faces = detector.detect_faces(frame)

    # Draw a rectangle around each detected face
    for result in faces:
        x, y, w, h = result['box']
        # Adjust the rectangle if it exceeds the frame boundaries
        frame_height, frame_width = frame.shape[:2]
        x_end = min(x + w, frame_width - 1)
        y_end = min(y + h, frame_height - 1)
        x_start = max(0, x)
        y_start = max(0, y)
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)

    # Display the frame with detected faces
    cv2.imshow('Face Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close the windows
cap.release()
cv2.destroyAllWindows()