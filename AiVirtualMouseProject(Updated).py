import cv2
import numpy as np
import HandTrackingModulei as htm
import time
import autopy
import pynput

##########################
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 7
##########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access the camera.")
    exit()

cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Error: Failed to capture image.")
            continue

        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
            fingers = detector.fingersUp()

            # Display operation name
            operation = ""

            # 1. Normal Gesture - Opened 5 Fingers
            if fingers == [1, 1, 1, 1, 1]:
                operation = "Normal Gesture"
                cv2.putText(img, operation, (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

            # 2. Move Cursor
            if fingers == [0, 1, 1, 0, 0]:
                operation = "Move Cursor"
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                autopy.mouse.move(wScr - clocX, clocY)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY

            # 3. Left Click
            if fingers == [0, 1, 0, 0, 0]:
                operation = "Left Click"
                autopy.mouse.click()

            # 4. Right Click
            if fingers == [0, 0, 1, 0, 0]:
                operation = "Right Click"
                autopy.mouse.click(button=autopy.mouse.Button.RIGHT)

            # 5. Double Click
            if fingers == [0, 1, 1, 0, 0]:
                length, img, _ = detector.findDistance(8, 12, img)
                if length < 40:
                    operation = "Double Click"
                    autopy.mouse.click()
                    time.sleep(0.1)
                    autopy.mouse.click()
            # 6. Scroll Up (Thumb and Index Finger Open)
            if fingers == [1, 1, 0, 0, 0]:
                mouse = pynput.mouse.Controller()
                operation = "Scroll Up"
                mouse.scroll(0, 2)  # Scroll up

            # 7. Scroll Down (Thumb and Middle Finger Open)
            if fingers == [1, 0, 1, 0, 0]:
                mouse = pynput.mouse.Controller()
                operation = "Scroll Down"
                mouse.scroll(0, -2)  # Scroll down

            # 8. Zoom In (Thumb Stretched, Others Closed)
            if fingers == [1, 0, 0, 0, 0]:
                mouse = pynput.mouse.Controller()
                operation = "Zoom In"
                mouse.scroll(0, 5)  # Simulate Zoom In by fast scrolling up

            # 9. Zoom Out (Thumb Closed, Others Open)
            if fingers == [0, 1, 1, 1, 1]:
                mouse = pynput.mouse.Controller()
                operation = "Zoom Out"
                mouse.scroll(0, -5)  # Simulate Zoom Out by fast scrolling down

            # Display operation name
            if operation:
                cv2.putText(img, operation, (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # Frame Rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        # Display
        cv2.imshow("Image", img)
        cv2.waitKey(1)

except KeyboardInterrupt:
    print("\nProcess interrupted by user.")
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Camera and resources released.")

