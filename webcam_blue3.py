
'''
    0 <------- pan left(pan++) ----- > 300   320   340 <----- pan right(pan--) -------> 640
  0 +-----------------------------------+-----+-----+------------------------------------+
    |                                   |     |     |                ^                   | 
    |                                   |     |     |                |                   | 
    |                                   |     |     |          tilt up(tilt--)           | 
    |                                   |     |     |                |                   | 
    |                                   |     |     |                v                   | 
220 +-----------------------------------+-----+-----+------------------------------------+ 
    |                                   |     |     |                                    | 
    |                                   |     |     |                                    |  
240 +-----------------------------------+-----+-----+------------------------------------+ 
    |                                   |     |     |                                    | 
    |                                   |     |     |                                    |  
260 +-----------------------------------+-----+-----+------------------------------------+ 
    |                                   |     |     |                ^                   | 
    |                                   |     |     |                |                   | 
    |                                   |     |     |          tilt down(tilt++)         | 
    |                                   |     |     |                |                   | 
    |                                   |     |     |                v                   |  
480 +-----------------------------------+-----+-----+------------------------------------+ 

'''
from getchar import Getchar
import cv2
import serial                                                     
import numpy as np
#sp  = serial.Serial('COM4', 9600, timeout=1)
sp  = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
webcam = cv2.VideoCapture(0)
#fps = int(webcam.get(cv2.CAP_PROP_FPS))
#print(f"Webcam FPS: {fps}")
#webcam.set(cv2.CAP_PROP_FPS, 60)  # 원하는 프레임 레이트로 설정 (예: 30fps)
#fps = int(webcam.get(cv2.CAP_PROP_FPS))
#print(f"Webcam FPS: {fps}")
#webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
#webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240) 

#webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
#webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 

pan = tilt = 90
_pan = _tilt = 90

margin_x = 20
margin_y = 20

def send_pan(pan):                     #def: 함수 선언
    tx_dat = "pan" + str(pan) + "\n"
    sp.write(tx_dat.encode())
    print(tx_dat)

def send_tilt(tilt):
    tx_dat = "tilt" + str(tilt) + "\n"
    sp.write(tx_dat.encode())
    print(tx_dat)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

def main(args=None):
    global pan; global _pan; global tilt; global _tilt;
    send_pan(90)
    send_tilt(90)
    kb = Getchar()
    key = ''
    
    while webcam.isOpened():
    
        status,img = webcam.read()
        frame =  cv2.flip(img,1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100,100,120])
        upper_blue = np.array([150,255,255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        res = cv2.bitwise_and(frame, frame, mask=mask)

        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        _, bin = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        
        largest_contour = None
        largest_area = 0    
        
        COLOR = (0, 255, 0)
        for cnt in contours:                # find largest blue object
            area = cv2.contourArea(cnt)
            if area > largest_area:
                largest_area = area
                largest_contour = cnt
                
         # draw bounding box with green line
        if largest_contour is not None:
            #area = cv2.contourArea(cnt)
            if largest_area > 500:  # draw only larger than 500
                x, y, width, height = cv2.boundingRect(largest_contour)       
                cv2.rectangle(frame, (x, y), (x + width, y + height), COLOR, 2)
                center_x = x + width//2
                center_y = y + height//2
                print("center: ( %s, %s )"%(center_x, center_y)) 
                
                if center_x < 320- margin_x:
                    print("pan left")
                    if pan - 1 >= 0:
                        pan = pan - 1
                        _pan = pan
                    else:
                        pan = 0
                        _pan = pan
                elif center_x > 320 + margin_x:
                    print("pan right")
                    if pan + 1 <= 180:
                        pan = pan + 1
                        _pan = pan
                    else:
                        pan = 180
                        _pan = pan
                else:
                    print("pan stop")
                    pan = _pan
                
                send_pan(pan) 
                    
                if center_y < 240- margin_y:
                    print("tilt down")
                    if tilt - 1 >= 0:
                        tilt = tilt - 1
                        _tilt = tilt
                    else:
                        tilt = 0
                        _tilt = tilt
                elif center_y > 240 + margin_y:
                    print("tilt up")
                    if tilt + 1 <= 180:
                        tilt = tilt + 1
                        _tilt = tilt
                    else:
                        tilt = 180
                        _tilt = tilt
                else:
                    print("tilt stop")
                    tilt = _tilt
                    
                send_tilt(tilt)
                
                #tx_dat = "pan" + str(pan) + "\n"
                #tx_dat = "tilt" + str(tilt) + "\n"
                #sp.write(tx_dat.encode())
                
        cv2.imshow("VideoFrame",frame)       # show original frame
        '''
        cv2.imshow('blue', res)           # show applied blue mask
        cv2.imwrite("blue.png", res)
        cv2.imshow('Green', res1)          # show applied green mask
        cv2.imwrite("green.png", res1)
        cv2.imshow('red', res2)          # show applied red mask
        cv2.imwrite("red.png", res2)
        '''
        k = cv2.waitKey(5) & 0xFF
            
        if k == 27:
            break
       
            
    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
