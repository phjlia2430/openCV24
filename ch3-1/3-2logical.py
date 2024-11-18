import cv2
import sys
import numpy as np

src1=cv2.imread('lenna512.png')
src2=cv2.imread('opencv_logo256.png')

if src1 is None or src2 is None:
    sys.exit('파일을 찾을 수 없습니다.')

mask = cv2.imread('opencv_logo256_mask.png',cv2.IMREAD_GRAYSCALE)
mask_inv = cv2.imread('opencv_logo256_mask_inv.png',cv2.IMREAD_GRAYSCALE)

#sy, sx = 100,100 #로고 위치 바꿀 수 있음
sy, sx = 0,0
rows,cols,channels = src2.shape #더 작은 이미지 크기 가져옴
roi = src1[sy:sy+rows, sx:sx+cols] #해당부분만큼 큰 이미지를 잘라냄
cv2.imshow('roi',roi)


src1_bg = cv2.bitwise_and(roi, roi, mask=mask) # mask의 흰색(1)에 해당하는 roi는 그대로, 검정색(0)은 검정색으로
cv2.imshow('src1_bg',src1_bg)

src2_fg = cv2.bitwise_and(src2, src2, mask=mask_inv) # mask_inv의 흰색(1)에 해당하는 src2는 그대로, 검정색(0)은 검정색으로
cv2.imshow('src2_fg',src2_fg)

dst = cv2.bitwise_or(src1_bg, src2_fg)
cv2.imshow('dst',dst)

src1[sy:sy+rows, sx:sx+cols] = dst #원래이미지에 dst 로고를 그대로 배치

pp=np.hstack((src1_bg,src2_fg, dst))
cv2.imshow('point processing - logical',pp)
cv2.imshow('combine', src1)

cv2.waitKey()
cv2.destroyAllWindows()