import cv2
import numpy as np

#cap = cv2.VideoCapture("ball.mp4")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Define yellow HSV range
lower = np.array([20, 100, 100])
upper = np.array([30, 255, 255])

# Set maximum display size (change if needed)
max_width = 800
max_height = 600

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---- Resize while keeping aspect ratio ----
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    frame = cv2.resize(frame, (new_w, new_h))

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mask for yellow
    mask = cv2.inRange(hsv, lower, upper)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        circle_area = np.pi * (radius ** 2)

        if 0.7 < area / circle_area < 1.3:
            center = (int(x), int(y))
            radius = int(radius)

            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            cv2.putText(frame, f"Centroid: {center}", (center[0] + 10, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Mask", mask)
    cv2.imshow("Ball Tracking", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()