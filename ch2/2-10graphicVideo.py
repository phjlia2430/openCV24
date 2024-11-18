import cv2
import sys
import numpy as np

cap = cv2.VideoCapture('slow_traffic_small.mp4') # 동영상을 가져오는 클래스
if not cap.isOpened():
    sys.exit('카메라 연결 실패')
    
while True:
    ret,frame=cap.read()
    if not ret:
        print('프레임 획득에 실패하여 루프를 나갑니다.')
        break

    pts = np.array([[180, 100], [190, 210], [300, 340]], dtype=np.int32)
    cv2.polylines(frame, [pts], False, (255, 0, 0), 10)
    cv2.line(frame, (400, 100), (640, 200), (0, 255, 0), 10)

    #random하게 원을 그리기
    #y = np.random.rand() * frame.shape[0] #rand() : 0-1사이 랜덤값
    #x = np.random.rand() * frame.shape[1]
    #cv2.circle(frame,(int(x),int(y)),20,(0,255,255),-1)

    h,w=frame.shape[:2]

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,100,200)

    linesp = cv2.HoughLinesP(edges, 1, np.pi/180, 10, None, minLineLength=50, maxLineGap=5)
    for line in linesp:  # 검출된 모든 선 순회
        x1, y1, x2, y2 = line[0]  # 시작점과 끝점
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('Video display', frame)
    
    key=cv2.waitKey(1)	# 1밀리초 동안 키보드 입력 기다림
    if key==ord('q'):	# 'q' 키가 들어오면 루프를 빠져나감
        break 
    
cap.release()			# 카메라와 연결을 끊음
cv2.destroyAllWindows()