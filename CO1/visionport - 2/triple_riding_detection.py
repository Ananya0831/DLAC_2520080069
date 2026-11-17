
import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open traffic video
video = cv2.VideoCapture("input/videos/traffic.mp4")

if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# Store unique motorcycle IDs flagged
triple_riding_vehicles = set()

print("Starting triple-riding detection...")
print("Press Q to stop.")

while True:

    success, frame = video.read()

    if not success:
        break

    # Detect people and motorcycles
    results = model.track(
        frame,
        persist=True,
        classes=[0, 3],
        tracker="bytetrack.yaml",
        verbose=False
    )

    motorcycles = []
    people = []

    if results[0].boxes is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.int().cpu().tolist()

        ids = results[0].boxes.id

        if ids is not None:
            ids = ids.int().cpu().tolist()
        else:
            ids = [-1] * len(boxes)

        for box, class_id, track_id in zip(boxes, classes, ids):

            x1, y1, x2, y2 = map(int, box)

            if class_id == 3:
                motorcycles.append(
                    (x1, y1, x2, y2, track_id)
                )

            elif class_id == 0:
                people.append(
                    (x1, y1, x2, y2)
                )

    # Count people associated with each motorcycle
    for bike in motorcycles:

        bx1, by1, bx2, by2, bike_id = bike

        rider_count = 0

        # Expand motorcycle region to include riders above it
        region_x1 = bx1 - 30
        region_y1 = by1 - int((by2 - by1) * 2.5)
        region_x2 = bx2 + 30
        region_y2 = by2 + 10

        for person in people:

            px1, py1, px2, py2 = person

            person_center_x = (px1 + px2) // 2
            person_center_y = (py1 + py2) // 2

            if (
                region_x1 <= person_center_x <= region_x2
                and region_y1 <= person_center_y <= region_y2
            ):
                rider_count += 1

        # Determine triple-riding status
        if rider_count > 2:

            status = "TRIPLE RIDING!"
            color = (0, 0, 255)

            if bike_id != -1:
                triple_riding_vehicles.add(bike_id)

        else:

            status = f"Riders: {rider_count}"
            color = (0, 255, 0)

        # Draw motorcycle bounding box
        cv2.rectangle(
            frame,
            (bx1, by1),
            (bx2, by2),
            color,
            2
        )

        cv2.putText(
            frame,
            status,
            (bx1, max(20, by1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # Display alert count
    cv2.putText(
        frame,
        f"Triple Riding Alerts: {len(triple_riding_vehicles)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.imshow("Triple Riding Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

print("\nTriple Riding Detection Report")
print("--------------------------------")
print("Vehicles flagged:", len(triple_riding_vehicles))
