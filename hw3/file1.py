import cv2
import numpy as np
import math

# 피부색 범위 설정
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# 입력 비디오 설정
cap = cv2.VideoCapture('video.mp4')  # 비디오 파일 경로 확인


if not cap.isOpened():
    print('동영상 연결 실패')
    exit()

frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),   # 비디오 크기 지정
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# 비디오 인코더 및 출력 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
output = cv2.VideoWriter('./record.mp4', fourcc, fps, frame_size)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('프레임 획득에 실패했습니다.')
        break

    # 프레임 좌우 반전 및 HSV 변환
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 피부색 영역 마스크 생성
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Gaussian Blur 적용
    mask = cv2.GaussianBlur(mask, (5,5), 0)

    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.erode(mask, None, iterations=1)

    # 윤곽선 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 가장 큰 윤곽선 선택
        hand_contour = max(contours, key=cv2.contourArea)

        # 손 윤곽선 그리기
        cv2.drawContours(frame, [hand_contour], -1, (255, 0, 0), 2)

        # Convex Hull과 Convexity Defects 계산
        hull = cv2.convexHull(hand_contour, returnPoints=False)
        defects = cv2.convexityDefects(hand_contour, hull)

        # 손가락 끝점 개수 계산
        finger_count = 0
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(hand_contour[s][0])
                end = tuple(hand_contour[e][0])
                far = tuple(hand_contour[f][0])

                # 손가락 끝점 각도 계산
                a = math.dist(start, end)
                b = math.dist(start, far)
                c = math.dist(end, far)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))

                # 깊이 기준과 각도 기준
                if d > 4000 and angle <= math.radians(100):
                    finger_count += 1
                    cv2.circle(frame, start, 5, (0, 0, 255), -1)

        # 가위바위보 판별
        if finger_count == 0:
            result = "Rock"
        elif finger_count == 2:
            result = "Scissors"
        elif finger_count >= 4:
            result = "Paper"
        else:
            result = "Undetermined"

        # 결과 표시
        cv2.putText(frame, result, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

    # 결과 비디오에 프레임 저장
    output.write(frame)


    # 결과 프레임을 출력
    cv2.imshow('Rock Paper Scissors', frame)

    # 일정 시간 대기 (프레임 간의 지연)
    if cv2.waitKey(70) & 0xFF == ord('q'):
        break


cap.release()
output.release()
cv2.destroyAllWindows()
