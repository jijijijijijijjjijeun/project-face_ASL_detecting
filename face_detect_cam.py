import cv2
import numpy as np

cam = cv2.VideoCapture(0)


# xml file path
cascade_filename = 'haarcascade_frontalface_alt.xml'
# load model
cascade = cv2.CascadeClassifier(cascade_filename)

if not cam.isOpened():
    print("Could not open webcam")
    exit()

def main(args=None):

    while cam.isOpened():

        status, org = cam.read()
        img = cv2.resize(org, dsize=(640, 480))
        #img = cv2.resize(frame, dsize=None, fx=1.0, fy=1.0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        results = cascade.detectMultiScale(gray,            # input image
                                               scaleFactor= 1.1,# image pyrmid factor
                                               minNeighbors=5,  # neighbor obj minimum distance(pixel)
                                               minSize=(20,20)  # minimum size of detected obj
                                               )
        for box in results:
                x, y, w, h = box
                print("center = (%s, %s)" %(x+w//2, y+h//2))
                
                cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,255), thickness=2)
                
        cv2.imshow('result',img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()