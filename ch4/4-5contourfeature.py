import cv2
import sys
import numpy as np

img=cv2.imread('star.png')
#img = cv2.imread('horse.png')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

h,w = img.shape[:2]
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
canny=cv2.Canny(gray,100,200)
#cv2.imshow('contours-canny',canny)

contours,hierarchy=cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

contour=contours[0]
cv2.drawContours(img,contour,-1,(255,0,255),2)
cv2.imshow('contours',img)

perimeter = cv2.arcLength(contour, closed=True) #둘레 길이 계산
print('perimeter=', perimeter)

area = cv2.contourArea(contour)
print('area=', area)

x, y, width, height = cv2.boundingRect(contour) #일반 사각형
#print(x,y,width,height)
#cv2.rectangle(img, (x, y), (x + width, y + height), (0, 0, 255), 2)

rect = cv2.minAreaRect(contour) # (중심, (가로,세로), 기울기)  -> 직사각형 중에서 면적이 최소화되는 사각형(기울어짐)
#print(rect) #센터,w,h,각도(기울어짐)
box = cv2.boxPoints(rect)   # 4개의 꼭지점 좌표
box = np.int32(box)     # 정수 좌표
#cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

(x, y), radius = cv2.minEnclosingCircle(contour)
#cv2.circle(img, (int(x), int(y)), int(radius), (255, 0, 0), 2)

poly10 = cv2.approxPolyDP(contour, epsilon=10, closed=True)
poly20 = cv2.approxPolyDP(contour, epsilon=20, closed=True)
poly50 = cv2.approxPolyDP(contour, epsilon=50, closed=True)
# 2 : epsilon. 다각형의 직선과의 허용 거리. 값이 크면 저장되는 좌표점의 개수가 작아짐
# 3 : closed?(닫힌 다각형?)
#epsilon이 작을 수록 윤곽선에 가까워짐
#cv2.drawContours(img, [poly10], 0, (0, 0, 255), 2)
#cv2.drawContours(img, [poly20], 0, (0, 255, 0), 2)
#cv2.drawContours(img, [poly50], 0, (255, 0, 0), 2)
print(contour.shape)
print(poly10.shape)
print(poly50.shape) #크면 저장되는 좌표점 개수가 줄어든다

for y in range(h):
    for x in range(w):
        if cv2.pointPolygonTest(contour, (x, y), False) > 0:
            img[y, x] = (255, 200, 255)

hull = cv2.convexHull(contour,returnPoints=True)
#cv2.drawContours(img, [hull], 0, (100, 100, 100), 2) #볼록다각형
#print(hull.shape)

hull2 = cv2.convexHull(contour, returnPoints=False) #False로 하면 볼록다각형에 대해서 결함되는 부분이 어딘지 정보를 구할 수 있음
defects = cv2.convexityDefects(contour, hull2)
print(defects) #하나 결함마다 4개값(시작,끝,먼곳,먼곳까지 거리)
s, e, f, d = defects[0, 0]
print(contour[s][0])
for i in range(defects.shape[0]):
    s, e, f, d = defects[i, 0]
    print(i,' : ', s,e,f,d)
    cv2.circle(img, tuple(contour[s][0]), 5, [128, 0, 255], -1)
    cv2.circle(img, tuple(contour[e][0]), 5, [255, 0, 128], -1)
    cv2.circle(img, tuple(contour[f][0]), 5, [128, 255, 0], -1)

cv2.imshow('contours',img)

cv2.waitKey()
cv2.destroyAllWindows()