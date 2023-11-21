from getchar import Getchar
import cv2
import serial  
import timeit  
import time        
sp  = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
# 가중치 파일 경로
cascade_filename = 'haarcascade_frontalface_alt.xml'
# 모델 불러오기
cascade = cv2.CascadeClassifier(cascade_filename)

cam = cv2.VideoCapture(2)

pan = 95
tilt = 85
_pan = 95
_tilt = 85

margin_x = 50
margin_y = 50

def send_pan(pan):  
    print("                       ",end ="")
    tx_dat = "pan" + str(pan) + "\n"
    sp.write(tx_dat.encode())
    print(tx_dat)

def send_tilt(tilt):
    print("                       ",end ="")
    tx_dat = "tilt" + str(tilt) + "\n"
    sp.write(tx_dat.encode())
    print(tx_dat)

if not cam.isOpened():
    print("Could not open webcam")
    exit()
# 영상 검출기
def videoDetector(cam,cascade):
       
    global pan; global _pan; global tilt; global _tilt;
    send_pan(95)
    send_tilt(85)
    kb = Getchar()
    key = ''
    
    while cam.isOpened():                                                   
       
        # 캡처 이미지 불러오기
        ret,img1 = cam.read()
        img = cv2.flip(img1, 1)
        # 영상 압축
        img = cv2.resize(img,dsize=None,fx=1.0,fy=1.0)
        # 그레이 스케일 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        # cascade 얼굴 탐지 알고리즘 
        results = cascade.detectMultiScale(gray,            # 입력 이미지
                                           scaleFactor= 1.1,# 이미지 피라미드 스케일 factor
                                           minNeighbors=5,  # 인접 객체 최소 거리 픽셀
                                           minSize=(20,20)  # 탐지 객체 최소 크기
                                           )
                                                                        
        for box in results:
            x, y, w, h = box
            center_x = x + w//2
            center_y = y + h//2
            print("center = (%s, %s)" %(center_x, center_y))
            if center_x < 320 - margin_x:
                print("                              ",end ="")
                print("pan left")
                if pan - 1 >= 0:
                    pan = pan - 1
                    _pan = pan
                else:
                    pan = 0
                    _pan = pan
            elif center_x > 320 + margin_x:
                print("                              ",end ="")
                print("pan right")
                if pan + 1 <= 180:
                    pan = pan + 1
                    _pan = pan
                else:
                    pan = 180
                    _pan = pan
            else:
                print("                              ",end ="")
                print("pan stop")
                pan = _pan
            
            send_pan(pan) 
                
            if center_y < 240 - margin_y:
                print("                              ",end ="")
                print("tilt down")
                if tilt - 1 >= 0:
                    tilt = tilt - 1
                    _tilt = tilt
                else:
                    tilt = 0
                    _tilt = tilt
            elif center_y > 240 + margin_y:
                print("                              ",end ="")
                print("tilt up")
                if tilt + 1 <= 180:
                    tilt = tilt + 1
                    _tilt = tilt
                else:
                    tilt = 180
                    _tilt = tilt
            else:
                print("                              ",end ="")
                print("tilt stop")
                tilt = _tilt
                    
            send_tilt(tilt)
            
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), thickness=2)
     

         # 영상 출력        
        cv2.imshow('VideoFrame',img)
        
        k = cv2.waitKey(5) & 0xFF
            
        if k == 27:
            break
       
            
    capture.release()
    cv2.destroyAllWindows()                                    
# 영상 탐지기
videoDetector(cam,cascade)                                                                                                                                                                                           
