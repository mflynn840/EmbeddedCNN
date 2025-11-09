#download raspery pi os lite
#extract the image file
#extract the device tree and kernel from the image files last 2 partitions

import cv2 #NOTE you have to uninstall neural_compressor and cv headless as they shadow opencv gui
import serial
import time


def overlayNumber(img, number):
    # overlay big red number
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 6          # adjust size
    thickness = 8           # thickness of the strokes
    color = (0,0,0)     # red in BGR
    org = (90, 200)         # (x, y) position of bottom-left corner of text

    cv2.putText(img, str(number), org, font, font_scale, color, thickness, cv2.LINE_AA)
    return img

def cameraLoop(serial_port = "COM3"):
    
    #ser = serial.Serial(serial_port, 115200) 
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Failed to open camera")

        
    while True:
        isNext, frame = cap.read()
        if not isNext:
            break
        
        #resize to 28x28x1 tensor
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (28, 28))
        
        #send to the ESP32
        #ser.write(small.tobytes())
        
        #await a response class from ESP32
        
        #display frame on screen with label
        cv2.imshow("Live Feed", frame)
        demo_frame = overlayNumber(cv2.resize(small, (280, 280)), 1)
        cv2.imshow("Gray 28x28", demo_frame)
        
        #exit display if q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time.sleep(0.1)  # ~10 fps
        
if __name__ == "__main__":
    cameraLoop()
