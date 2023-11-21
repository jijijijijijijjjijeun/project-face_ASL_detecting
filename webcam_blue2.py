'''
    0 <------ pan left(pan++) ------- > 300   320 <------ pan right(pan--) ------------> 640
  0 +------------------------------------+-----+-----+------------------------------------+
    |                                    |     |     |                ^                   | 
    |                                    |     |     |                |                   | 
    |                                    |     |     |          tilt up(tilt--)           | 
    |                                    |     |     |                |                   | 
    |                                    |     |     |                v                   | 
220 +------------------------------------+-----+-----+------------------------------------+ 
    |                                    |     |     |                                    | 
    |                                    |     |     |                                    |  
240 +------------------------------------+-----+-----+------------------------------------+ 
    |                                    |     |     |                                    | 
    |                                    |     |     |                                    |  
260 +------------------------------------+-----+-----+------------------------------------+ 
    |                                    |     |     |                ^                   | 
    |                                    |     |     |                |                   | 
    |                                    |     |     |                                    | 
    |                                    |     |     |          tilt down(tilt++)         | 
    |                                    |     |     |                |                   | 
    |                                    |     |     |                v                   |  
480 +------------------------------------+-----+-----+------------------------------------+ 

'''
import cv2
import serial
import numpy as np
#sp = serial.Serial('COM4',9600,timeout = 1)  윈도우
sp  = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)

margin_x =int(20)
margin_y =int(20)
cam_center_x = 320
cam_center_y = 240

pos_x = 90
pos_y = 90

_pos_x = 90
_pos_y = 90

webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

while webcam.isOpened():
    status, frame = webcam.read()
    if not status:
        break

    frame_flipped = cv2.flip(frame, 1)  # 좌우 반전

    hsv = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 100, 120])
    upper_blue = np.array([150, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res = cv2.bitwise_and(frame_flipped, frame_flipped, mask=mask)

    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    _, bin = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    largest_area = 0

    COLOR = (0, 255, 0)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > largest_area:
            largest_area = area
            largest_contour = cnt

    if largest_contour is not None:
        if largest_area > 500:
            x, y, width, height = cv2.boundingRect(largest_contour)
            cv2.rectangle(frame_flipped, (x, y), (x + width, y + height), COLOR, 2)
            center_x = x + width // 2
            center_y = y + height // 2
            print("center: ( %s, %s )" % (center_x, center_y))
            if center_x > 300 and center_x < 340 and center_y > 220 and center_y < 260:
                print("stop")
            else:
                if center_x < cam_center_x - margin_x :
                    print("pan right : ", end=' ')
                    print(cam_center_x - center_x)
                    if pos_x + 1 <= 640:
                        pos_x = pos_x + 1
                        _pos_x = pos_x
                    else:
                        pos_x = 640
                        _pos_x = pos_x
                elif center_x > cam_center_x + margin_x:
                    print("pan left : ", end=' ')
                    print(center_x - cam_center_x)
                    if pos_x - 1 >= 0:
                        pos_x = pos_x - 1
                        _pos_x = pos_x
                    else:
                        pos_x = 0
                        _pos_x = pos_x
                else:
                    print("pan stop")
                    pos_x = _pos_x
                if center_y < cam_center_y - margin_y:
                    print("tilt down : ", end=' ')
                    print(cam_center_y - center_y)
                    if pos_y + 1 <= 480:
                        pos_y = pos_y + 1
                        _pos_y = pos_y
                    else:
                        pos_y = 480
                        _pos_y = pos_y
                elif center_y > cam_center_y + margin_y:
                    print("tilt up : ", end=' ')
                    print(center_y - cam_center_y)
                    if pos_y - 1 >= 0:
                        pos_y = pos_y - 1
                        _pos_y = pos_y
                    else:
                        pos_y = 0
                        _pos_y = pos_y
                else:
                    print("tilt stop")
                    pos_y = _pos_y
         
                tx_dat = "pan" + str(pos_x)     #sp.write로 전송
                sp.write(tx_dat
                .encode())

    cv2.imshow("VideoFrame", frame_flipped)

    k = cv2.waitKey(5) & 0xFF

    if k == 27:
        break

webcam.release()
cv2.destroyAllWindows()
