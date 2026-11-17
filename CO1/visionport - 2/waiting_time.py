
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
    fps = video.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    # Store previous vehicle positions
    previous_positions = {}

    # Store time when vehicles stop
    stopped_since = {}

    # Define stationary movement threshold
    movement_threshold = 3

    # Minimum time to classify as waiting
    waiting_threshold = 1.5

    frame_number = 0

    print("Starting vehicle waiting time estimation...")
    print("Press Q to stop.")

    while True:
        ret, frame = video.read()

        if not ret:
            break

        frame_number += 1

        # Track vehicles
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

                    current_time = frame_number / fps

                    status = "MOVING"
                    waiting_time = 0

                    if track_id in previous_positions:

                        old_x, old_y = previous_positions[track_id]

                        # Calculate movement
                        movement = math.sqrt(
                            (center_x - old_x) ** 2 +
                            (center_y - old_y) ** 2
                        )

                        if movement < movement_threshold:

                            if track_id not in stopped_since:
                                stopped_since[track_id] = current_time

                            waiting_time = (
                                current_time - stopped_since[track_id]
                            )

                            if waiting_time >= waiting_threshold:
                                status = "WAITING"

                        else:
                            stopped_since.pop(track_id, None)

                    previous_positions[track_id] = current_position

                    # Display vehicle status
                    if status == "WAITING":
                        label = (
                            f"ID {track_id} WAITING: "
                            f"{waiting_time:.1f}s"
                        )
                        color = (0, 0, 255)

                    else:
                        label = f"ID {track_id} MOVING"
                        color = (0, 255, 0)

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

        cv2.imshow(
            "VisionPort - Vehicle Waiting Time",
            frame
        )

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

print("Vehicle waiting time estimation completed!")
