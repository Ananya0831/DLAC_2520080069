
import cv2
import os

# TASK 3: Read and display a traffic image

print("Opening traffic image...")

image = cv2.imread("input/images/traffic.jpg")

if image is None:
    print("Image not found! Check the file path.")
else:
    cv2.imshow("Traffic Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# TASK 2: Capture an image using webcam

print("Starting webcam...")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Unable to access camera.")

else:
    print("Press S to capture an image.")
    print("Press Q to close the camera.")

    while True:
        ret, frame = camera.read()

        if not ret:
            print("Unable to read camera.")
            break

        cv2.imshow("Traffic Camera", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            cv2.imwrite(
                "input/images/captured_traffic.jpg",
                frame
            )
            print("Image saved successfully!")

        elif key == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


# TASK 4: Load and play a traffic video

print("Opening traffic video...")

video = cv2.VideoCapture("input/videos/traffic.mp4")

if not video.isOpened():
    print("Unable to open traffic video.")

else:
    print("Playing traffic video...")
    print("Press Q to stop video playback.")

    while True:
        ret, frame = video.read()

        if not ret:
            print("Video playback completed.")
            break

        cv2.imshow("Traffic Video", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


# TASK 5: Extract individual frames from traffic video

print("Extracting video frames...")

video = cv2.VideoCapture("input/videos/traffic.mp4")

os.makedirs("output/frames", exist_ok=True)

if not video.isOpened():
    print("Unable to open video for frame extraction.")

else:
    fps = video.get(cv2.CAP_PROP_FPS)

    frame_interval = max(1, round(fps))

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = video.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = f"output/frames/frame_{saved_count}.jpg"

            cv2.imwrite(filename, frame)

            saved_count += 1

        frame_count += 1

    video.release()

    print("Frame extraction completed!")
    print("Total frames extracted:", saved_count)

print("Program completed.")
