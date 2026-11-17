
from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open traffic video
video = cv2.VideoCapture("input/videos/traffic.mp4")

if not video.isOpened():
    print("Unable to open traffic video.")

else:
    # Vehicle class IDs
    vehicle_classes = [2, 3, 5, 7]

    # Define region of interest
    roi = np.array([
        [150, 150],
        [500, 150],
        [500, 450],
        [150, 450]
    ], dtype=np.int32)

    # Congestion threshold
    congestion_threshold = 10

    while True:
        ret, frame = video.read()

        if not ret:
            break

        # Detect vehicles
        results = model(frame, verbose=False)

        vehicle_count = 0

        for result in results:
            for box in result.boxes:

                class_id = int(box.cls[0])

                if class_id in vehicle_classes:

                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0]
                    )

                    # Calculate bounding box centre
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # Check if vehicle is inside ROI
                    if cv2.pointPolygonTest(
                        roi,
                        (center_x, center_y),
                        False
                    ) >= 0:

                        vehicle_count += 1

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

        # Draw ROI
        cv2.polylines(
            frame,
            [roi],
            True,
            (255, 0, 0),
            2
        )

        # Determine congestion
        if vehicle_count >= congestion_threshold:
            status = "CONGESTION DETECTED"
            color = (0, 0, 255)

        else:
            status = "TRAFFIC NORMAL"
            color = (0, 255, 0)

        # Display information
        cv2.putText(
            frame,
            "Vehicles in Junction: " + str(vehicle_count),
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            status,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        cv2.imshow("VisionPort - Congestion Monitoring", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

print("Congestion monitoring completed!")
