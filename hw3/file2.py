import cv2
import numpy as np

car_no = str(input("자동차 영상 번호(00~05): "))
img = cv2.imread('cars/' + car_no + '.jpg')
if img is None:
    print("이미지를 읽을 수 없습니다. 파일 경로를 확인하세요.")
    exit()

def preprocess_image(img):
    """ 1단계: 전처리 (블러링) """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)  # Gaussian 블러링 적용
    return blurred

def edge_detection(image):
    """ 세로 엣지 검출 (Prewitt 필터 사용) """
    prewitt_filter_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    prewitt_grad_x = cv2.filter2D(image, -1, prewitt_filter_x)
    return cv2.convertScaleAbs(prewitt_grad_x)

def thresholding(image):
    """ 임계값을 이용한 에지 분리 """
    _, th = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def close_operation(image):
    """ 닫힘 연산을 통한 번호판 영역 검출 """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)

def verify_aspect_size(size):
    """ 종횡비와 크기 확인 """
    w, h = size
    if h == 0 or w == 0:
        return False
    aspect = h / w if h > w else w / h
    chk1 = 3000 < (h * w) < 12000 # 번호판 넓이 조건
    chk2 = 2.0 < aspect < 6.5       # 번호판 종횡비 조건
    return chk1 and chk2

# 번호판 후보 검출
def find_license_plate_candidates(image):
    preprocessed = preprocess_image(image)
    edges = edge_detection(preprocessed)
    thresh = thresholding(edges)
    closed = close_operation(thresh)

    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        size = rect[1]

        if verify_aspect_size(size):
            candidates.append(box)

    return candidates, closed, contours

# 후보 검출
candidates, closed_image, contours = find_license_plate_candidates(img)

cv2.imshow('Original Image', img)

cv2.imshow('Closed Operation Image', closed_image)

# 윤곽선을 찾고 최소 면적 사각형을 그린 이미지 시각화
img_with_contours = img.copy()  # 원본 이미지를 복사하여 윤곽선 표시
for contour in contours:
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.polylines(img_with_contours, [box], isClosed=True, color=(255, 0, 0), thickness=2)  # 윤곽선 그리기

cv2.imshow('Contours with Min Area Rectangle', img_with_contours)

# 최종 후보 시각화
final_image = img.copy()  # 최종 이미지도 원본을 복사하여 후보 표시
for candidate in candidates:
    cv2.polylines(final_image, [candidate], isClosed=True, color=(0, 255, 0), thickness=2)

cv2.imshow('Final Candidates', final_image)

# 결과 대기
cv2.waitKey(0)
cv2.destroyAllWindows()
