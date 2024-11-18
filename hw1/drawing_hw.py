import cv2
import numpy as np

# 흰색 배경 (600x900 크기)
canvas = np.ones((600, 900, 3), dtype=np.uint8) * 255
drawing = False  # 그리기 여부 확인
ix, iy = -1, -1  # 초기 좌표
shape_type = None  # 현재 그리는 도형 타입

def draw_shape(event, x, y, flags, param):
    global ix, iy, drawing, canvas, shape_type

    # 마우스 왼쪽 버튼 누름
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
        if flags & cv2.EVENT_FLAG_ALTKEY:  # Alt + Left: 직사각형 그리기 시작
            drawing = True
            shape_type = 'rectangle'
        elif flags & cv2.EVENT_FLAG_CTRLKEY:  # Ctrl + Left: 원 그리기 시작
            drawing = True
            shape_type = 'circle'
        elif flags & cv2.EVENT_FLAG_SHIFTKEY:  # Shift + Left: 초록색 원 그리기
            drawing = True
            shape_type = 'shift_left'
        else:  # 그냥 왼쪽 버튼: 파란색 원
            cv2.circle(canvas, (x, y), 5, (255, 0, 0), -1)

    # 마우스 오른쪽 버튼 누름
    elif event == cv2.EVENT_RBUTTONDOWN:
        ix, iy = x, y
        if flags & cv2.EVENT_FLAG_ALTKEY:  # Alt + Right: 내부 채운 직사각형 시작
            drawing = True
            shape_type = 'filled_rectangle'
        elif flags & cv2.EVENT_FLAG_CTRLKEY:  # Ctrl + Right: 내부 채운 원 시작
            drawing = True
            shape_type = 'filled_circle'
        elif flags & cv2.EVENT_FLAG_SHIFTKEY:  # Shift + Right: 노란색 원 그리기
            drawing = True
            shape_type = 'shift_right'
        else:  # 그냥 오른쪽 버튼: 빨간색 원
            cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1)

    # 마우스 움직임
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_canvas = canvas.copy()
            if shape_type == 'rectangle':  # 직사각형
                cv2.rectangle(temp_canvas, (ix, iy), (x, y), (0, 0, 255), 2)
            elif shape_type == 'filled_rectangle':  # 내부 채운 직사각형
                cv2.rectangle(temp_canvas, (ix, iy), (x, y), (0, 0, 255), -1)
            elif shape_type == 'circle':  # 원
                radius = int(np.sqrt((x - ix) ** 2 + (y - iy) ** 2))
                cv2.circle(temp_canvas, (ix, iy), radius, (0, 255, 0), 2)
            elif shape_type == 'filled_circle':  # 내부 채운 원
                radius = int(np.sqrt((x - ix) ** 2 + (y - iy) ** 2))
                cv2.circle(temp_canvas, (ix, iy), radius, (0, 255, 0), -1)
            elif shape_type == 'shift_left':  # 초록색 원
                cv2.circle(canvas, (x, y), 5, (0, 255, 0), -1)  # 초록색 원을 실제 캔버스에 바로 그리기
            elif shape_type == 'shift_right':  # 노란색 원
                cv2.circle(canvas, (x, y), 5, (0, 255, 255), -1)  # 노란색 원을 실제 캔버스에 바로 그리기
            cv2.imshow('Paint', temp_canvas)

        # 그냥 마우스 움직일 때 (도형 그리기 없이)
        else:
            if flags & cv2.EVENT_FLAG_LBUTTON:  # 파란색 원
                cv2.circle(canvas, (x, y), 5, (255, 0, 0), -1)
            elif flags & cv2.EVENT_FLAG_RBUTTON:  # 빨간색 원
                cv2.circle(canvas, (x, y), 5, (0, 0, 255), -1)

    # 마우스 버튼 뗐을 때
    elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
        drawing = False
        if shape_type == 'rectangle':  # 직사각형
            cv2.rectangle(canvas, (ix, iy), (x, y), (0, 0, 255), 2)
        elif shape_type == 'filled_rectangle':  # 내부 채운 직사각형
            cv2.rectangle(canvas, (ix, iy), (x, y), (0, 0, 255), -1)
        elif shape_type == 'circle':  # 원
            radius = int(np.sqrt((x - ix) ** 2 + (y - iy) ** 2))
            cv2.circle(canvas, (ix, iy), radius, (0, 255, 0), 2)
        elif shape_type == 'filled_circle':  # 내부 채운 원
            radius = int(np.sqrt((x - ix) ** 2 + (y - iy) ** 2))
            cv2.circle(canvas, (ix, iy), radius, (0, 255, 0), -1)

        shape_type = None  # 도형 초기화

# 창과 마우스 이벤트 설정
cv2.namedWindow('Paint')
cv2.setMouseCallback('Paint', draw_shape)

# 키보드 입력 처리
while True:
    cv2.imshow('Paint', canvas)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # 's' 키를 누르면 이미지 저장
        cv2.imwrite('my_drawing.png', canvas)
        print("Image saved as 'my_drawing.png'")
    elif key == ord('q'):  # 'q' 키를 누르면 종료
        break

cv2.destroyAllWindows()
