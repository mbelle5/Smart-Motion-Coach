
import cv2

# Open your MacBook's webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open webcam.")
else:
    print("Webcam opened successfully!")

    while True:
        success, frame = camera.read()

        if not success:
            print("Failed to capture video.")
            break

        cv2.imshow("Smart Motion Coach - Camera Test", frame)

        # Press Q to close the camera
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
