
import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open traffic video
video = cv2.VideoCapture("input/videos/traffic.mp4")

if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# Vehicle class IDs: car, motorcycle, bus, truck
vehicle_classes = [2, 3, 5, 7]

# Store previous vertical positions of vehicles
previous_positions = {}

# Store vehicles that have already been counted
counted_vehicles = set()

# Entry and exit counters
entering_count = 0
leaving_count = 0

# Get video dimensions
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Horizontal counting line
line_y = height // 2

print("Starting vehicle entry and exit counting...")
print("Press Q to stop.")

while True:

    success, frame = video.read()

    if not success:
        break

    # Track vehicles using ByteTrack
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=vehicle_classes,
        verbose=False
    )

    # Draw counting line
    cv2.line(
        frame,
        (0, line_y),
        (width, line_y),
        (255, 255, 0),
        2
    )

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.int().cpu().tolist()
        classes = results[0].boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, ids, classes):

            x1, y1, x2, y2 = map(int, box)

            # Calculate vehicle centre
            center_y = (y1 + y2) // 2
            center_x = (x1 + x2) // 2

            # Vehicle label
            vehicle_name = model.names[class_id]

            # Check whether vehicle has crossed the line
            if track_id in previous_positions:

                previous_y = previous_positions[track_id]

                if track_id not in counted_vehicles:

                    # Top to bottom = entering
                    if previous_y < line_y and center_y >= line_y:
                        entering_count += 1
                        counted_vehicles.add(track_id)

                    # Bottom to top = leaving
                    elif previous_y > line_y and center_y <= line_y:
                        leaving_count += 1
                        counted_vehicles.add(track_id)

            # Update vehicle position
            previous_positions[track_id] = center_y

            # Draw bounding box and tracking ID
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{vehicle_name} ID:{track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

            # Draw vehicle centre
            cv2.circle(
                frame,
                (center_x, center_y),
                4,
                (0, 0, 255),
                -1
            )

    # Display entry and exit counts
    cv2.putText(
        frame,
        f"Entering: {entering_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Leaving: {leaving_count}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.imshow("Vehicle Entry and Exit Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

print("\nVehicle Counting Report")
print("-----------------------")
print("Vehicles Entering:", entering_count)
print("Vehicles Leaving:", leaving_count)
print("Total Vehicles Counted:", entering_count + leaving_count)
