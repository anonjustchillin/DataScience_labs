"""
Розробити програмний скрипт мовою Python що реалізує обчислювальний алгоритм
машинного навчання (Machine Learning (ML)) відповідно до технічних умов:

Підрахувати кількість об’єктів на обраному цифровому зображенні. Об’єкти, що
підлягають обрахунку обрати самостійно. Зміст етапів попередньої обробки зображень
(корекція кольору, фільтрація, векторизація, кластеризація) має буди результатом R&D
процесів, що конкретизується обраним зображенням і об’єктами для підрахунку. Провести
аналіз отриманих результатів, сформувати висновки.
"""

import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
from fontTools.merge import cmap

from MyImage import MyImage

FILENAME = './desk.jpg'
RESULT_FILE = './result.jpg'

def process_kmeans(img, k):
    Z = np.float32(img.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2

def normalize_hist(img):
    R, G, B = cv2.split(img)
    eq_R = cv2.equalizeHist(R)
    eq_G = cv2.equalizeHist(G)
    eq_B = cv2.equalizeHist(B)
    img_eq = cv2.merge((eq_R, eq_G, eq_B))
    return img_eq

def get_light_objects(img: MyImage, name='img_light'):
    gamma = 1.6
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    gamma_corrected = cv2.LUT(img.processed_img, lookUpTable)
    img.update_process_image(gamma_corrected, f'Gamma correction (gamma={gamma}, {name})')
    img.show_image(version=1)

    k = 6
    kmeans_img = process_kmeans(img.processed_img, k)
    img.update_process_image(kmeans_img, f'K-means (k={k}, {name})')
    img.show_image(version=1)

    gray = cv2.cvtColor(img.processed_img, cv2.COLOR_BGR2GRAY)
    img.update_process_image(gray, f'Grey Image ({name})')
    img.show_image(version=1)

    blurred = cv2.GaussianBlur(img.processed_img, (5, 5), 0)
    img.update_process_image(blurred, f'Gaussian Blur 2 ({name})')
    img.show_image(version=1)

    thresh_light = cv2.threshold(img.processed_img, 200, 255, cv2.THRESH_BINARY)[1]
    thresh_light = cv2.erode(thresh_light, None, iterations=7)
    thresh_light = cv2.dilate(thresh_light, (3, 3), iterations=0)
    img.update_process_image(thresh_light, f'Threshold ({name})')
    img.show_image(version=1)

    blurred = cv2.GaussianBlur(img.processed_img, (11, 11), 0)
    img.update_process_image(blurred, f'Gaussian Blur 3 ({name})')
    img.show_image(version=1)

    canny = cv2.Canny(img.processed_img, 0, 10, 3)
    img.update_process_image(canny, f'Canny ({name})')
    img.show_image(version=1)

    return img

def get_dark_objects(img: MyImage, name='img_dark'):
    gamma = 1.3
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    gamma_corrected = cv2.LUT(img.processed_img, lookUpTable)
    img.update_process_image(gamma_corrected, f'Gamma correction (gamma={gamma}, {name})')
    img.show_image(version=1)

    k = 4
    kmeans_img = process_kmeans(img.processed_img, k)
    img.update_process_image(kmeans_img, f'K-means (k={k}, {name})')
    img.show_image(version=1)

    gray = cv2.cvtColor(img.processed_img, cv2.COLOR_BGR2GRAY)
    img.update_process_image(gray, f'Grey Image ({name})')
    img.show_image(version=1)

    blurred = cv2.GaussianBlur(img.processed_img, (19, 19), 0)
    img.update_process_image(blurred, f'Gaussian Blur 2 ({name})')
    img.show_image(version=1)

    inverted = cv2.bitwise_not(img.processed_img)
    thresh_dark = cv2.threshold(inverted, 170, 255, cv2.THRESH_BINARY)[1]
    thresh_dark = cv2.erode(thresh_dark, None, iterations=7)
    thresh_dark = cv2.dilate(thresh_dark, (3, 3), iterations=3)
    img.update_process_image(thresh_dark, f'Threshold ({name})')
    img.show_image(version=1)

    canny = cv2.Canny(img.processed_img, 0, 10, 3)
    img.update_process_image(canny, f'Canny ({name})')
    img.show_image(version=1)

    return img

if __name__ == '__main__':
    print('Вхідне зображення')
    img = MyImage(FILENAME)
    img2 = MyImage(FILENAME)
    img.show_image()
    print(img.original_img)

    print()
    print('Фільтрація зображення')

    img_eq = normalize_hist(img.original_img)
    img.update_process_image(img_eq, 'Histogram Normalization')
    img.show_image(version=1)

    blurred = cv2.GaussianBlur(img.processed_img, (9, 9), 0)
    img.update_process_image(blurred, 'Gaussian Blur')
    img2.update_process_image(blurred, 'Gaussian Blur')
    img.show_image(version=1)

    img_light = get_light_objects(img)
    img_dark = get_dark_objects(img2)
    combined = cv2.bitwise_or(img_light.processed_img, img_dark.processed_img)
    img.update_process_image(combined, 'Bitwise OR (img_light, img_dark)')
    img.show_image(version=1)

    print()
    print("Прорахунок об'єктів на зображенні")

    min_area = 15
    all_cnts, _ = cv2.findContours(img.processed_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [cnt for cnt in all_cnts if cv2.contourArea(cnt) > min_area]

    rgb = cv2.cvtColor(img.original_img, cv2.COLOR_BGR2RGB)
    cv2.drawContours(rgb, cnts, -1, (0, 255, 0), 2)

    plt.imshow(rgb)
    plt.title(f'Total objects found: {len(cnts)}')
    plt.savefig(RESULT_FILE)
    plt.show()
