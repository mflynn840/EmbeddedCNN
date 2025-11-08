#download raspery pi os lite
#extract the image file
#extract the device tree and kernel from the image files last 2 partitions

import cv2
import serial
import time

ser = serial.Serial("COM3", 115200)  # Replace COM3 with your ESP32 port

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (28, 28))
    ser.write(small.tobytes())
    time.sleep(0.1)  # ~10 fps
