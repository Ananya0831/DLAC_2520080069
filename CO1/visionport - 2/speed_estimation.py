
from ultralytics import YOLO
import cv2
import math

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open traffic video
video = cv2.VideoCapture("input/videos/traffic.mp4")

if not video.isOpened():
    print("Unable to open traffic video.")

else:
    print("Starting vehicle speed estimation...")
    print("Press Q to stop.")

    fps = video.get(cv2.CAP_PROP_FPS)

    # Assumed real-world distance in metres
    known_distance = 10

    # Track previous positions
    previous_positions = {}

    # Assumed pixel distance corresponding to known distance
    pixel_distance = 100

    while True:
        ret, frame = video.read()

        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[2, 3, 5, 7],
            verbose=False
        )

        if results[0].boxes is not None:
            boxes = results[0].boxes

            if boxes.id is not None:

                track_ids = boxes.id.int().cpu().tolist()
                coordinates = boxes.xyxy.cpu().numpy()

                for track_id, box in zip(track_ids, coordinates):

                    x1, y1, x2, y2 = map(int, box)

                    # Calculate vehicle centre
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    current_position = (center_x, center_y)

                    if track_id in previous_positions:

                        old_position, old_frame = previous_positions[track_id]

                        pixel_movement = math.sqrt(
                            (center_x - old_position[0]) ** 2 +
                            (center_y - old_position[1]) ** 2
                        )

                        frame_difference = (
                            video.get(cv2.CAP_PROP_POS_FRAMES) - old_frame
                        )

                        if frame_difference > 0 and pixel_distance > 0:

                            time_taken = frame_difference / fps

                            distance_moved = (
                                pixel_movement / pixel_distance
                            ) * known_distance

                            speed = (
                                distance_moved / time_taken
                            ) * 3.6

                            label = f"ID {track_id} Speed: {speed:.1f} km/h"

                            cv2.putText(
                                frame,
                                label,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2
                            )

                    previous_positions[track_id] = (
                        current_position,
                        video.get(cv2.CAP_PROP_POS_FRAMES)
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

        cv2.imshow("VisionPort - Speed Estimation", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

print("Speed estimation completed!")
