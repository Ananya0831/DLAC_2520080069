
from ultralytics import YOLO
import cv2

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Read traffic image
image = cv2.imread("input/images/traffic.jpg")

if image is None:
    print("Image not found!")

else:
    # Detect vehicles
    results = model(image)

    vehicle_classes = {
        2: "Cars",
        3: "Motorcycles",
        5: "Buses",
        7: "Trucks"
    }

    vehicle_count = {
        "Cars": 0,
        "Motorcycles": 0,
        "Buses": 0,
        "Trucks": 0
    }

    # Count detected vehicles
    for result in results:
        for box in result.boxes:

            class_id = int(box.cls[0])

            if class_id in vehicle_classes:
                name = vehicle_classes[class_id]
                vehicle_count[name] += 1

        result.show()

    # Calculate total vehicles
    total_vehicles = sum(vehicle_count.values())

    # Display vehicle count
    print("\n========== VEHICLE COUNT ==========")

    for vehicle, count in vehicle_count.items():
        print(vehicle, ":", count)

    print("-----------------------------------")
    print("Total Vehicles:", total_vehicles)

    # TASK 10: Calculate traffic density
    if total_vehicles <= 5:
        density = "LOW"
        status = "NORMAL"

    elif total_vehicles <= 15:
        density = "MEDIUM"
        status = "MODERATE TRAFFIC"

    else:
        density = "HIGH"
        status = "CONGESTED"

    print("\n========== TRAFFIC ANALYSIS ==========")
    print("Traffic Density:", density)
    print("Traffic Status :", status)
    print("======================================")

print("Traffic analysis completed!")
