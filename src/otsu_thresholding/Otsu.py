import numpy as np
import cv2
from typing import Tuple

def open_image_as_grayscale(image_name: str) -> np.ndarray:
    img = cv2.imread(image_name)
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayscale_img.astype(np.int32)

def create_histogram(image: np.ndarray) -> np.ndarray:
    COLOR_RANGE = 255
    image_array = image.flatten()
    number_of_pixels = image_array.size
    histogram = np.zeros((COLOR_RANGE + 1,), dtype=int)
    for i in image_array:
        histogram[i] += 1
    return histogram / number_of_pixels

def calculate_mean(probabilities: np.ndarray) -> np.float32:
    mean = 0.
    for i in range(probabilities.size):
        mean += i * probabilities[i]
    return np.float32(mean)

def calculate_variance(mean: np.float32, probabilities: np.ndarray) -> np.float32:
    variance = 0.
    for i in range(probabilities.size):
        variance += (i - mean)**2 * probabilities[i]
    return np.float32(variance)

def search_optimal_threshold(variance: np.float32, histogram: np.ndarray) -> Tuple[np.float32, np.float32]:
    final_threshold = -1
    final_value = -1
    for t in range(1,histogram.size):
        C1 = histogram[:t]
        C2 = histogram[t:]
        u1, o1, w1 = within_class_variance(C1)
        u2, o2, w2 = within_class_variance(C2)
        Wt = w1*o1 + w2*o2 #Within class variance for t
        Bt = variance - Wt #Between class variance for t
        if Bt > final_value:
            final_value = Bt
            final_threshold = t
    return (final_value, final_threshold)

def within_class_variance(histogram_part: np.ndarray) -> Tuple[np.float32, np.float32, np.float32]:
    weight = np.float32(histogram_part.sum())
    mean = calculate_mean(probabilities= histogram_part) / weight
    variance = calculate_variance(mean= mean, probabilities= histogram_part) / weight
    return (mean, variance, weight)

def create_binarized_image(image: np.ndarray, threshold: np.int) -> np.ndarray:
    binarized = image.copy()
    binarized[image >= threshold] = 255
    binarized[image < threshold] = 0
    return binarized

def main():
    image_name = 'julia.png'
    loaded_image = open_image_as_grayscale(image_name= image_name)
    histogram = create_histogram(image= loaded_image)
    histogram_mean = calculate_mean(probabilities= histogram)
    histogram_variance = calculate_variance(mean= histogram_mean, probabilities= histogram)
    value, threshold = search_optimal_threshold(variance= histogram_variance, histogram= histogram)
    binarized_image = create_binarized_image(image= loaded_image, threshold= threshold)
    cv2.imwrite(image_name.split('.')[0] + '_threshold_at' + str(threshold) + '.png',binarized_image.astype(np.uint8))
main()