
import cv2
from ultralytics import YOLO
from collections import defaultdict, deque

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open traffic video
video = cv2.VideoCapture("input/videos/traffic.mp4")

if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# Vehicle classes
vehicle_classes = [2, 3, 5, 7]

# Store recent positions of each vehicle
vehicle_history = defaultdict(lambda: deque(maxlen=12))

# Count unique vehicles flagged
wrong_way_vehicles = set()

# Number of consecutive wrong-way checks required
wrong_way_frames = defaultdict(int)

# Assumed lane directions:
# Left half: Top to Bottom
# Right half: Bottom to Top
LEFT_LANE_DIRECTION = "down"
RIGHT_LANE_DIRECTION = "up"

# Movement threshold in pixels
movement_threshold = 12

print("Starting wrong-way vehicle detection...")
print("Press Q to stop.")

while True:

    success, frame = video.read()

    if not success:
        break

    height, width = frame.shape[:2]
    lane_divider = width // 2

    # Track vehicles
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=vehicle_classes,
        verbose=False
    )

    # Draw lane divider
    cv2.line(
        frame,
        (lane_divider, 0),
        (lane_divider, height),
        (255, 255, 0),
        2
    )

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, ids):

            x1, y1, x2, y2 = map(int, box)

            # Calculate vehicle centre
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Store centre position
            vehicle_history[track_id].append(
                (center_x, center_y)
            )

            status = "CHECKING"
            color = (0, 255, 255)

            if len(vehicle_history[track_id]) >= 8:

                positions = vehicle_history[track_id]

                first_y = positions[0][1]
                last_y = positions[-1][1]

                movement = last_y - first_y

                # Determine expected direction from lane
                if center_x < lane_divider:
                    expected_direction = LEFT_LANE_DIRECTION
                    lane_name = "LEFT LANE"
                else:
                    expected_direction = RIGHT_LANE_DIRECTION
                    lane_name = "RIGHT LANE"

                # Check movement against expected direction
                wrong_direction = False

                if movement_threshold < abs(movement):

                    if expected_direction == "down" and movement < 0:
                        wrong_direction = True

                    elif expected_direction == "up" and movement > 0:
                        wrong_direction = True

                if wrong_direction:
                    wrong_way_frames[track_id] += 1
                else:
                    wrong_way_frames[track_id] = 0

                # Confirm only after repeated checks
                if wrong_way_frames[track_id] >= 5:
                    status = "WRONG WAY!"
                    color = (0, 0, 255)
                    wrong_way_vehicles.add(track_id)
                else:
                    status = lane_name
                    color = (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Display vehicle ID and status
            cv2.putText(
                frame,
                f"ID:{track_id} {status}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

    # Display wrong-way count
    cv2.putText(
        frame,
        f"Wrong Way Vehicles: {len(wrong_way_vehicles)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.imshow("Wrong-Way Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

print("\nWrong-Way Detection Report")
print("---------------------------")
print("Vehicles flagged:", len(wrong_way_vehicles))
