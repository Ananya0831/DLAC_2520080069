
import cv2
from ultralytics import YOLO
from collections import defaultdict

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open traffic video
video = cv2.VideoCapture("input/videos/traffic.mp4")

if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# Vehicle classes: car, motorcycle, bus, truck
vehicle_classes = [2, 3, 5, 7]

# Store previous positions
previous_positions = {}

# Store movement history
movement_history = defaultdict(list)

# Track alerted vehicles
alerted_vehicles = set()

# Expected direction:
# Change to "up" if most vehicles move bottom to top.
expected_direction = "down"

print("Starting wrong-direction detection...")
print("Press Q to stop.")

while True:

    success, frame = video.read()

    if not success:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=vehicle_classes,
        verbose=False
    )

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, ids):

            x1, y1, x2, y2 = map(int, box)

            # Calculate vehicle centre
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Store current position
            movement_history[track_id].append(center_y)

            # Keep only recent positions
            if len(movement_history[track_id]) > 10:
                movement_history[track_id].pop(0)

            status = "MOVING"

            if track_id in previous_positions:

                previous_y = previous_positions[track_id]

                movement = center_y - previous_y

                # Ignore very small movements
                if abs(movement) > 5:

                    if expected_direction == "down" and movement < 0:
                        status = "WRONG DIRECTION"
                        alerted_vehicles.add(track_id)

                    elif expected_direction == "up" and movement > 0:
                        status = "WRONG DIRECTION"
                        alerted_vehicles.add(track_id)

            previous_positions[track_id] = center_y

            # Display detection
            if status == "WRONG DIRECTION":

                color = (0, 0, 255)

            else:

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
                f"ID:{track_id} {status}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

    # Display total alerts
    cv2.putText(
        frame,
        f"Wrong Direction Alerts: {len(alerted_vehicles)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.imshow("Wrong Direction Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

print("\nWrong Direction Detection Report")
print("--------------------------------")
print("Vehicles flagged:", len(alerted_vehicles))
