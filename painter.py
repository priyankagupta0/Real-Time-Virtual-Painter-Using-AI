import cv2
import numpy as np
import time
import os
import handtrackingmodule as htm

brushThickness = 25
eraserThickness = 100

folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f"{folderPath}/{imPath}")
    overlayList.append(image)
# print(len(overlayList))
header = overlayList[0]
drawColor = (255, 0, 255)
xp, yp = 0, 0

cap = cv2.VideoCapture(0)
detector = htm.handDetector(detectionCon=0.65, maxHands=1)
imgCanvas = np.zeros((480, 640, 3), np.uint8)
while True:
    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)
    # img = cv2.resize(img, (1280, 720))

    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    x1, y1, x2, y2 = 0, 0, 0, 0
    if len(lmList) != 0:
        # print(lmList)
        # tip of index and middle fingers
        # if(len(lmList)>=13):
        nested_list = lmList[0]
        if len(nested_list)>=8:
            x1, y1 = nested_list[8][1], nested_list[8][2]
            x2, y2 = nested_list[12][1], nested_list[12][2]
        # print(x1,y1)

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        # 4. If Selection Mode - Two finger are up
        if len(fingers) > 0:
            if fingers[1] and fingers[2]:
                xp, yp = 0, 0
                # print("Selection Mode")
                if y1 < 100:
                    if 190 < x1 < 260:
                        header = overlayList[0]
                        drawColor = (255, 0, 255)
                    elif 260 < x1 < 360:
                        header = overlayList[1]
                        drawColor = (0, 255, 0)
                    elif 360 < x1 < 460:
                        header = overlayList[2]
                        drawColor = (255, 0, 0)
                    elif 460 < x1 < 590:
                        header = overlayList[3]
                        drawColor = (0, 0, 0)
                cv2.rectangle(
                    img, (x1, y1 - 15), (x2, y2 + 15), drawColor, cv2.FILLED
                )

            if fingers[1] and fingers[2] == False:

                cv2.circle(img, (x1, y1),10, drawColor, cv2.FILLED)
                # print("Drawing Mode")
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                if drawColor == (0, 0, 0):
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            
                else:
                    cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                    cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
                xp, yp = x1, y1
    else:
        # Handle the case when no fingers are up
        pass

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)
    # Setting the header image
    img[0:100, 0:640] = header
    img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    # cv2.imshow("Inv", imgInv)
    cv2.waitKey(1)
