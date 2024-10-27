from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

import os

model = YOLO("yolov11n-face.pt")
camera = cv2.VideoCapture(0)
img_counter = 0
font_path_linux = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
font_path_windows = "C:\\Windows\\Fonts\\segoeui.ttf" 

if os.path.exists(font_path_linux):
    font = ImageFont.truetype(font_path_linux, 14)
else:
    font = ImageFont.truetype(font_path_windows, 14)

while True:
    ret, frame = camera.read()

    if not ret:
        break

    results = model(frame)
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
        
            confidence = box.conf[0]
            label = f"top programmer {confidence:.2f}"
            draw.text((x1, y1 - 25), label, font=font, fill=(0, 255, 0)) 

    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("mem", frame)

    img_path = f"path/opencv_frame_{img_counter}.png"
    cv2.imwrite(img_path, frame)
    img_counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()