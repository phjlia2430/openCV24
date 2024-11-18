import cv2
import sys

from numpy.ma.core import resize

img = cv2.imread('girl_laughing.jpg')
if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

x, y, w, h = cv2.selectROI(img) 	# 관심 영역을 선택
print("Selected ROI:", x, y, w, h)

roi = img[y:y+h, x:x+w]	# 선택한 영역만 잘라냄

#ch3 블러 적용해보기
#mosaic:3-2 smoothing
#blur_roi = cv2.blur(roi,(25,25))
#img[y:y+h, x:x+w] = blur_roi

#mosaic:3-3 resize
resize_roi = cv2.resize(roi,(w//15,h//15))
resize_roi = cv2.resize(resize_roi,(w,h),interpolation=cv2.INTER_NEAREST) #축소했다가 다시 확대
img[y:y+h, x:x+w] = resize_roi


#cv2.imwrite('roi.jpg', roi)	# 잘라낸 영역을 저장

#cv2.imshow('ROI', roi)
cv2.imshow('ROI-Blur', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
