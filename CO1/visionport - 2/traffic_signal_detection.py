
import cv2
import numpy as np

# Open traffic video
video = cv2.VideoCapture("input/videos/traffic.mp4")

if not video.isOpened():
    print("Unable to open traffic video.")

else:
    # Read first frame
    ret, frame = video.read()

    if not ret:
        print("Unable to read video.")
        video.release()

    else:
        # Select traffic signal region
        print("Select the traffic light region and press ENTER.")

        roi = cv2.selectROI(
            "Select Traffic Signal",
            frame,
            False
        )

        cv2.destroyWindow("Select Traffic Signal")

        x, y, w, h = roi

        if w == 0 or h == 0:
            print("No region selected.")
            video.release()

        else:
            # Restart video
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # HSV colour ranges
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])

            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])

            lower_yellow = np.array([15, 100, 100])
            upper_yellow = np.array([35, 255, 255])

            lower_green = np.array([40, 50, 50])
            upper_green = np.array([90, 255, 255])

            print("Starting traffic signal detection...")
            print("Press Q to stop.")

            while True:
                ret, frame = video.read()

                if not ret:
                    break

                # Extract selected region
                signal = frame[y:y+h, x:x+w]

                # Convert to HSV
                hsv = cv2.cvtColor(
                    signal,
                    cv2.COLOR_BGR2HSV
                )

                # Detect red
                red_mask1 = cv2.inRange(
                    hsv, lower_red1, upper_red1
                )

                red_mask2 = cv2.inRange(
                    hsv, lower_red2, upper_red2
                )

                red_mask = cv2.bitwise_or(
                    red_mask1, red_mask2
                )

                # Detect yellow
                yellow_mask = cv2.inRange(
                    hsv, lower_yellow, upper_yellow
                )

                # Detect green
                green_mask = cv2.inRange(
                    hsv, lower_green, upper_green
                )

                # Count colour pixels
                red_count = cv2.countNonZero(red_mask)
                yellow_count = cv2.countNonZero(yellow_mask)
                green_count = cv2.countNonZero(green_mask)

                # Determine signal status
                colour_counts = {
                    "RED": red_count,
                    "YELLOW": yellow_count,
                    "GREEN": green_count
                }

                detected_colour = max(
                    colour_counts,
                    key=colour_counts.get
                )

                if colour_counts[detected_colour] < 5:
                    status = "SIGNAL NOT DETECTED"
                    display_colour = (255, 255, 255)

                elif detected_colour == "RED":
                    status = "RED - STOP"
                    display_colour = (0, 0, 255)

                elif detected_colour == "YELLOW":
                    status = "YELLOW - CAUTION"
                    display_colour = (0, 255, 255)

                else:
                    status = "GREEN - GO"
                    display_colour = (0, 255, 0)

                # Draw selected region
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    (255, 0, 0),
                    2
                )

                # Display signal status
                cv2.putText(
                    frame,
                    status,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    display_colour,
                    2
                )

                cv2.imshow(
                    "VisionPort - Traffic Signal Detection",
                    frame
                )

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            video.release()
            cv2.destroyAllWindows()

print("Traffic signal detection completed!")
