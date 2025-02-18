import cv2
import numpy as np
import sys

def onCornerHarris(thresh):
    img = img_copy.copy()
    CN = cv2.normalize(C, 0, 400, cv2.NORM_MINMAX)  # 트랙바는 정수만 다룸
    rcorners = []
    for j in range(1, C.shape[0] - 1):  # 비최대 억제
        for i in range(1, C.shape[1] - 1):
            if CN[j, i] > thresh and sum(sum(CN[j, i] > CN[j - 1:j + 2, i - 1:i + 2])) == 8:  # 주변 8개와 비교해서 [ji]가 더 큰 값이 8이면, 모든 이웃보다 큰값을 가지면
                rcorners.append((i, j))

    for pt in rcorners:
        cv2.circle(img, pt, 3, (255, 0, 255), -1)  # 좌표 표시
    print("임계값: %2d , 코너 개수: %2d" % (thresh, len(rcorners)))
    cv2.imshow("harris detect", img)

img = cv2.imread('shapes4.png', cv2.IMREAD_COLOR)
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

img_copy = img.copy()

median = cv2.medianBlur(img, 3)
gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)

blockSize = 4  # 이웃 화소 범위
apertureSize = 3  # 소벨 마스크 크기
k = 0.04
C = cv2.cornerHarris(gray, blockSize, apertureSize, k)  # OpenCV 제공 함수

# 1 C값에 대한 thresholding
#img[C>0.05*C.max()]=[0,0,255]
#print(C.max(),0.05*C.max())
#cv2.imshow('harris detect', img)

# 2 C값에 대한 트랙바
thresh = 5  # 코너 응답 임계값
onCornerHarris(thresh)
cv2.createTrackbar("Threshold", "harris detect", thresh, 30, onCornerHarris)
# ("트랙 바 이름", "윈도우 창 제목", 현재값 변수, 최댓값, 콜백 함수)

cv2.waitKey()
cv2.destroyAllWindows()