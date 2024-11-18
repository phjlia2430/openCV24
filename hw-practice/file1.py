import cv2
import numpy as np

# 처리할 이미지 파일 목록
items = ['items/item1.jpg', 'items/item2.jpg', 'items/item3.jpg', 'items/item4.jpg']
output_folder = 'items/'

img2 = cv2.imread('items/items.jpg')
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


sift = cv2.SIFT_create()
kp2, des2 = sift.detectAndCompute(gray2, None)
flann_matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
T = 0.7

# 각 item에 대해 반복 수행
for i, item_path in enumerate(items, start=1):
    img1 = cv2.imread(item_path)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    kp1, des1 = sift.detectAndCompute(gray1, None)

    knn_match = flann_matcher.knnMatch(des1, des2, 2)

    # 좋은 매칭 선택
    good_match = []
    for nearest1, nearest2 in knn_match:
        if (nearest1.distance / nearest2.distance) < T:
            good_match.append(nearest1)

    # 매칭된 특징점으로 homography 계산
    points1 = np.float32([kp1[gm.queryIdx].pt for gm in good_match])
    points2 = np.float32([kp2[gm.trainIdx].pt for gm in good_match])
    H, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

    # homography가 적용된 위치 계산
    h1, w1 = img1.shape[:2]
    box1 = np.float32([[0, 0], [0, h1 - 1], [w1 - 1, h1 - 1], [w1 - 1, 0]]).reshape(4, 1, 2)
    box2 = cv2.perspectiveTransform(box1, H)

    # img2에 다각형으로 매칭 위치 그리기
    img_with_box = img2.copy()  # 원본 이미지를 수정하지 않도록 복사
    img_with_box = cv2.polylines(img_with_box, [np.int32(box2)], True, (0, 255, 0), 8)

    # 매칭 결과 이미지 생성
    img_match = np.empty((max(h1, img2.shape[0]), w1 + img2.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(img1, kp1, img_with_box, kp2, good_match, img_match,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 화면 크기에 맞게 이미지 축소
    screen_width = 800
    screen_height = 600
    img_height, img_width = img_match.shape[:2]
    scale_w = screen_width / img_width
    scale_h = screen_height / img_height
    scale = min(scale_w, scale_h)
    resized_dim = (int(img_width * scale), int(img_height * scale))
    resized_img_match = cv2.resize(img_match, resized_dim, interpolation=cv2.INTER_AREA)

    # 결과 이미지 저장
    output_path = f"{output_folder}match_item{i}.jpg"
    cv2.imwrite(output_path, resized_img_match)
    print(f"Saved result for {item_path} as {output_path}")

    # 이미지 표시
    cv2.imshow(f'Matches and Homography - Item {i}', resized_img_match)

cv2.waitKey(0)
cv2.destroyAllWindows()
