import cv2

cap = cv2.VideoCapture('video.mp4')
current_effect = 'n'

while True:  # 무한루프
    ret, frame = cap.read()  # 비디오를 구성하는 프레임 획득(frame)
    if not ret:
        print('비디오 재생이 끝났습니다.')
        break

    key = cv2.waitKey(1)

    # 효과 미리 적용
    bila = cv2.bilateralFilter(frame, -1, 10, 5)
    sty = cv2.stylization(frame, sigma_s=60, sigma_r=0.45)
    graySketch, colorSketch = cv2.pencilSketch(frame, sigma_s=60, sigma_r=0.7, shade_factor=0.02)
    oil = cv2.xphoto.oilPainting(frame, 7, 1)

    # 키 입력 처리
    if key == ord('n'):
        current_effect = 'n'
    elif key == ord('b'):
        current_effect = 'b'
    elif key == ord('s'):
        current_effect = 's'
    elif key == ord('g'):
        current_effect = 'g'
    elif key == ord('c'):
        current_effect = 'c'
    elif key == ord('o'):
        current_effect = 'o'
    elif key == ord('q'):
        break

    # 효과에 따른 이미지 표시
    if current_effect == 'n':
        cv2.putText(frame, "Original", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 2)
        cv2.imshow('Special Effects', frame)
    elif current_effect == 'b':
        cv2.putText(bila, "Bilateral", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 2)
        cv2.imshow('Special Effects', bila)
    elif current_effect == 's':
        cv2.putText(sty, "Stylization", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 2)
        cv2.imshow('Special Effects', sty)
    elif current_effect == 'g':
        cv2.putText(graySketch, "GrayPencil", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (128, 128, 128), 2)
        cv2.imshow('Special Effects', graySketch)
    elif current_effect == 'c':
        cv2.putText(colorSketch, "ColorPencil", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 2)
        cv2.imshow('Special Effects', colorSketch)
    elif current_effect == 'o':
        cv2.putText(oil, "OilPainting", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 2)
        cv2.imshow('Special Effects', oil)

cap.release()
cv2.destroyAllWindows()
