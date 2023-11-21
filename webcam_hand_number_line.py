from getchar import Getchar
import cv2
import serial
import timeit
import time
import mediapipe as mp
import math

#sp  = serial.Serial('com7', 9600, timeout=1)
sp  = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
# 가중치 파일 경로
cascade_filename = 'haarcascade_frontalface_alt.xml'
# 모델 불러오기
cascade = cv2.CascadeClassifier(cascade_filename)

# Mediapipe 손 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mpDraw = mp.solutions.drawing_utils

# HAND_CONNECTIONS를 직접 정의
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # 손가락 각 관절 연결
    (0, 5), (5, 6), (6, 7), (7, 8),  # 엄지 손가락 연결
    (0, 9), (9, 10), (10, 11), (11, 12),  # 검지 손가락 연결
    (0, 13), (13, 14), (14, 15), (15, 16),  # 중지 손가락 연결
    (0, 17), (17, 18), (18, 19), (19, 20)  # 약지 손가락 연결
]

# tuple의 리스트로 변환
HAND_CONNECTIONS = [tuple(connection) for connection in HAND_CONNECTIONS]

def dist(x1, y1, x2, y2):
    return math.sqrt(math.pow(x1 - x2, 2)) + math.sqrt(math.pow(y1 - y2, 2))

compareIndex = [[18, 4], [6, 8], [10, 12], [14, 16], [18, 20]]
open = [False, False, False, False, False] 
gesture = [[False, False, False, False, False, "zero"],     #x
            [False, True, False, False, False, "one"], 
            [False, True, True, False, False, "two"],
            [False, True, True, True, False, "three"],
            [False, True, True, True, True, "four"],
            [True, True, True, True, True, "five"],
            [True, False, False, False, False, "six"],
            [True, True, False, False, False, "seven"],
            [True, True, True, False, False, "eight"],
            [True, True, True, True, False, "nine"] ]

cam = cv2.VideoCapture(2)

pan = 85
tilt = 75
_pan = 85
_tilt = 75

margin_x = 20
margin_y = 20

def send_pan(pan):
    print("                       ", end="")
    tx_dat = "pan" + str(pan) + "\n"
    sp.write(tx_dat.encode())
    print(tx_dat)

def send_tilt(tilt):
    print("                       ", end="")
    tx_dat = "tilt" + str(tilt) + "\n"
    sp.write(tx_dat.encode())
    print(tx_dat)

if not cam.isOpened():
    print("Could not open webcam")
    exit()
    

def videoDetector(cam, cascade):

    global pan
    global _pan
    global tilt
    global _tilt

    send_pan(93)
    send_tilt(70)
    kb = Getchar()
    key = ''

    while cam.isOpened():

        ret, img = cam.read()
        
        # 좌우 반전
        img = cv2.flip(img, 1)

        # 얼굴 검출
        img = cv2.resize(img, dsize=None, fx=1.0, fy=1.0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )


        for box in results:
            x, y, w, h = box
            center_x = x + w // 2
            center_y = y + h // 2
            print("center = (%s, %s)" % (center_x, center_y))
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
                print("                              ", end="")
                print("tilt down")
                if tilt - 1 >= 0:
                    tilt = tilt - 1
                    _tilt = tilt
                else:
                    tilt = 0
                    _tilt = tilt
            elif center_y > 240 + margin_y:
                print("                              ", end="")
                print("tilt up")
                if tilt + 1 <= 180:
                    tilt = tilt + 1
                    _tilt = tilt
                else:
                    tilt = 180
                    _tilt = tilt
            else:
                print("                              ", end="")
                print("tilt stop")
                tilt = _tilt

            send_tilt(tilt)

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), thickness=2)
        

        h, w, c = img.shape
        # 손 감지
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # 손가락 랜드마크 추출
                hand_landmarks = landmarks.landmark

                # 손가락 랜드마크 좌표 저장
                hand_points = []
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)  # Y 축 반전 제거
                    hand_points.append((cx, cy))

                # 손가락 랜드마크를 점으로 표시
                for point in hand_points:
                    cv2.circle(img, point, 5, (50, 50, 50), -1)

                # 다섯 손가락을 선으로 연결하여 표시
                fingers = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
                for finger in fingers:
                    for i in range(4):
                        start_point = hand_points[finger[i]]
                        end_point = hand_points[finger[i + 1]]
                        cv2.line(img, start_point, end_point, (100, 100, 100), 2)

                for i in range(0, 5):
                    open[i] = dist(landmarks.landmark[0].x, landmarks.landmark[0].y,
                              landmarks.landmark[compareIndex[i][0]].x, landmarks.landmark[compareIndex[i][0]].y) < \
                              dist(landmarks.landmark[0].x, landmarks.landmark[0].y,
                              landmarks.landmark[compareIndex[i][1]].x, landmarks.landmark[compareIndex[i][1]].y)
                #print(open)
                text_x = (landmarks.landmark[0].x * w)
                text_y = (landmarks.landmark[0].y * h)
                
             
                    
                for i in range(0, len(gesture)):
                    flag = True
                    for j in range(0, 5):
                        if gesture[i][j] != open[j]:
                            flag = False
                    if flag:
                        #img = cv2.flip(img, 1)
                        cv2.putText(img,gesture[i][-1], (round(text_x) , round(text_y) - 300),
                                    cv2.FONT_HERSHEY_SIMPLEX , 2, (255, 204, 204), 4)    

        # 영상 출력
        cv2.imshow('VideoFrame', img)

        k = cv2.waitKey(5) & 0xFF

        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

# 영상 탐지기
videoDetector(cam, cascade)
