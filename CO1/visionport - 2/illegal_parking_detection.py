
import cv2
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open traffic video
video = cv2.VideoCapture("input/videos/traffic.mp4")

if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# Vehicle classes: car, motorcycle, bus, truck
vehicle_classes = [2, 3, 5, 7]

# Get first frame
success, first_frame = video.read()

if not success:
    print("Error: Could not read video.")
    video.release()
    exit()

# Select restricted parking area
print("Select the restricted parking area.")
print("Drag a rectangle and press ENTER.")

roi = cv2.selectROI(
    "Select Restricted Parking Area",
    first_frame,
    False
)

cv2.destroyWindow("Select Restricted Parking Area")

x, y, w, h = map(int, roi)

if w == 0 or h == 0:
    print("No restricted area selected.")
    video.release()
    exit()

# Define restricted area
roi_x1 = x
roi_y1 = y
roi_x2 = x + w
roi_y2 = y + h

# Reset video to beginning
video.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Store vehicle positions and stationary start times
vehicle_positions = {}
stationary_since = {}

# Store vehicles already reported
illegal_parked = set()

# Parking threshold in seconds
parking_threshold = 5

# Movement threshold in pixels
movement_threshold = 5

# Video FPS
fps = video.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

frame_number = 0

print("Starting illegal parking detection...")
print("Press Q to stop.")

while True:

    success, frame = video.read()

    if not success:
        break

    frame_number += 1

    # Calculate video timestamp
    current_time = frame_number / fps

    # Track vehicles
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=vehicle_classes,
        verbose=False
    )

    # Draw restricted area
    cv2.rectangle(
        frame,
        (roi_x1, roi_y1),
        (roi_x2, roi_y2),
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        "Restricted Parking Area",
        (roi_x1, max(20, roi_y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
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

            # Check if vehicle centre is inside restricted area
            inside_roi = (
                roi_x1 <= center_x <= roi_x2
                and roi_y1 <= center_y <= roi_y2
            )

            status = "MOVING"
            color = (0, 255, 0)

            if inside_roi:

                if track_id not in vehicle_positions:

                    vehicle_positions[track_id] = (
                        center_x,
                        center_y
                    )

                    stationary_since[track_id] = current_time

                else:

                    previous_x, previous_y = vehicle_positions[track_id]

                    movement = (
                        (center_x - previous_x) ** 2
                        + (center_y - previous_y) ** 2
                    ) ** 0.5

                    if movement > movement_threshold:

                        # Vehicle is moving
                        stationary_since[track_id] = current_time

                    else:

                        # Check how long vehicle has remained stationary
                        waiting_time = (
                            current_time - stationary_since[track_id]
                        )

                        if waiting_time >= parking_threshold:

                            status = "ILLEGAL PARKING!"
                            color = (0, 0, 255)

                            illegal_parked.add(track_id)

                        else:

                            status = f"Stopped: {waiting_time:.1f}s"
                            color = (0, 255, 255)

                    vehicle_positions[track_id] = (
                        center_x,
                        center_y
                    )

            else:

                # Vehicle has left the restricted area
                vehicle_positions.pop(track_id, None)
                stationary_since.pop(track_id, None)

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

    # Display illegal parking count
    cv2.putText(
        frame,
        f"Illegal Parking Alerts: {len(illegal_parked)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.imshow("Illegal Parking Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

print("\nIllegal Parking Detection Report")
print("--------------------------------")
print("Vehicles flagged:", len(illegal_parked))
