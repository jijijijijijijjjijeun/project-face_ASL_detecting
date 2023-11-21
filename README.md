# project-object_detecting
얼굴인식후 화면가운데로 정렬하기 위해 서브모터(pan,tilt)움직임
손(MAX = 1)의 움직임을 인식하여 수화(ASL)로 대화가능

getchar.py : 리눅스
getcharr.py : 윈도

webcam_blue2.py : 파란색인식 (파란색물건이 화면에 잡히면 물건의 센터로 부터 높이와 넒이를 구해 박스 침, 박스의 센터값과 움직여야하는 값 출력)
webcam_blue3.py : 파란색 인식후 서브모터(아두이노로 컨트롤)에 부착된 카메라가 움직여 파란물체를 화면 가운데로 정렬

haarcascade_frontalface_alt.xml : 얼굴(눈, 코, 입)인식 xml파일
face_detect_cam.py : 내장cam으로 얼굴인식
webcam_face.py : 얼굴인식후 서브모터로 화면 가운데로 정렬
webcam_hand_number_line.py : 화면에 손(MAX = 1)이 잡히면 손가락의 개수로 숫자 출력(0 - 10) 

------------------최종-------------------
webcam_asl_knn : 화면에 얼굴을 트래킹하면서 손의 움직임(벡터와 벡터 사이의 각도- KNN학습)과 매핑된 수화(ASL)출력



