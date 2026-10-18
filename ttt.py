import cv2
import numpy as np

pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

green_lower = np.array([30, 60, 90], dtype=np.int8)
green_upper = np.array([230, 115, 180], dtype=np.int8)

kernel = np.ones((5,5), dtype=np.unit8)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    mask = cv2.inRange(lab_frame, green_lower, green_upper)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) > 500:
            (center_x, center_y), radius = cv2.minEnclosingCircle(largest_contour)

            center = (int(center_x),int(center_y)) 

            cv2.circle(frame, center, int(radius), (0, 255, 0), 2)
            cv2.circle(frame, center, 5 , (0, 0, 255), -1)

    cv2.imshow("Object Detection", frame)
    cv2.imshow("Biary Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

