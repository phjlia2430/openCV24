import cv2
import numpy as np
import os

# 자동차 이미지가 저장된 폴더 경로
image_folder = '/CVintro24/hw2/cars/'

# 이미지 목록
image_files = ['00.jpg', '01.jpg', '02.jpg', '03.jpg', '04.jpg', '05.jpg']

def preprocess_image(image):
    """ 1단계: 전처리 (블러링) """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(gray, (5, 5))  # 블러링 적용
    return blurred

def edge_detection(image):
    """ 2단계: 세로 엣지 검출 (Prewitt 필터 사용) """
    prewitt_filter_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])  # 수직 에지 필터
    prewitt_grad_x = cv2.filter2D(image, -1, prewitt_filter_x)  # 수직 에지 검출
    prewitt_x = cv2.convertScaleAbs(prewitt_grad_x)  # 절대값으로 변환하여 양수로
    return prewitt_x

def thresholding(image):
    """ 3단계: 임계값을 이용한 에지 분리 """
    _, th = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def close_operation(image):
    """ 4단계: 닫힘 연산을 통한 번호판 영역 검출 """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))  # 가로로 긴 구조 요소 생성
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)  # 원상태로 약간 침식
    return closed


def resize_image(image, scale_percent):
    """ 이미지 크기 조절 """
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def find_license_plate_candidates(image):
    """ 전체 과정 """
    # 각 단계별로 결과를 저장
    preprocessed = preprocess_image(image)      # 1단계 전처리
    edges = edge_detection(preprocessed)        # 2단계 세로 엣지 검출
    thresh = thresholding(edges)                # 3단계 임계값 적용
    closed = close_operation(thresh)            # 4단계 닫힘 연산

    # 이미지 축소 (예: 50% 크기로 축소)
    preprocessed_resized = resize_image(preprocessed, 50)
    edges_resized = resize_image(edges, 50)
    thresh_resized = resize_image(thresh, 50)
    closed_resized = resize_image(closed, 50)

    # 결과를 가로로 붙여 한 창에 표시
    combined = np.hstack([preprocessed_resized, edges_resized, thresh_resized, closed_resized])

    return combined


for file_name in image_files:
    image_path = os.path.join(image_folder, file_name)
    image = cv2.imread(image_path)

    result = find_license_plate_candidates(image)

    # 결과 출력 (각 단계별 이미지를 한 창에 띄움)
    cv2.imshow(f'Result - {file_name}', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
