
from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open traffic video
video = cv2.VideoCapture("input/videos/traffic.mp4")

if not video.isOpened():
    print("Unable to open traffic video.")

else:
    print("Starting vehicle tracking...")
    print("Press Q to stop tracking.")

    # Vehicle class IDs
    vehicle_classes = [2, 3, 5, 7]

    while True:
        ret, frame = video.read()

        if not ret:
            break

        # Track vehicles using ByteTrack
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=vehicle_classes,
            verbose=False
        )

        for result in results:

            if result.boxes is not None and result.boxes.id is not None:

                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().tolist()
                class_ids = result.boxes.cls.int().cpu().tolist()

                for box, track_id, class_id in zip(
                    boxes, track_ids, class_ids
                ):

                    x1, y1, x2, y2 = map(int, box)

                    # Get vehicle name
                    name = model.names[class_id]

                    # Draw bounding box
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # Display tracking ID and vehicle name
                    label = f"{name} ID: {track_id}"

                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

        cv2.imshow("VisionPort - Vehicle Tracking", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

print("Vehicle tracking completed!")
