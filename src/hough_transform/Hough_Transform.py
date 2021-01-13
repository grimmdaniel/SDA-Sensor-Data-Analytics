#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
from math import sqrt, pi, cos, sin
from Prewitt_Edge_Detector import edge_detection
from Otsu import otsu_thresholding
from typing import Tuple, List


def open_image_as_grayscale(image_name: str) -> np.ndarray:
    img = cv2.imread(image_name)
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayscale_img.astype(np.int32)

def preprocess_image(grayscale_img: np.ndarray) -> np.ndarray:
    image_with_edges_detected = edge_detection(grayscale_image= grayscale_img)
    binary_image = otsu_thresholding(image_with_edges_detected)
    return binary_image

def generate_predifined_circlepoints(radius_min: int, radius_max: int) -> List[Tuple[float,float,float]]:
    steps = 100
    points = []
    for r in range(radius_min, radius_max + 1):
        for t in range(steps):
            points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))
    return points

def generate_accumulation_matrix(input_image: np.ndarray, points: List[Tuple[float,float,float]]) -> np.ndarray:
    accumulation_array = np.zeros(input_image.shape)
    height, width = input_image.shape
    for i in range(0,height):
        for j in range(0,width):
            if input_image[i][j] > 0: #it is white
                for r, dx, dy in points:
                    a = i - dx
                    b = j - dy
                    try:
                        accumulation_array[a,b] += 100
                    except IndexError:
                        pass
    accumulation_array = accumulation_array / np.max(accumulation_array) * 255
    return accumulation_array

def remove_noise_from_accumulator(accumulator_arr: np.ndarray, threshold: float) -> np.ndarray:
    min_acc_val = int(255 * threshold)
    height, width = accumulator_arr.shape
    for i in range(0,height):
        for j in range(0,width):
            if accumulator_arr[i,j] <= min_acc_val:
                accumulator_arr[i,j] = 0
    return accumulator_arr
    
def generate_circles(accumulator_arr: np.ndarray, distance_threshold: int) -> List[Tuple[int,int,float]]:
    circles = []
    height, width = accumulator_arr.shape
    for i in range(0,height):
        for j in range(0,width):
            if accumulator_arr[i,j] > 0:
                current_value = accumulator_arr[i,j]
                larger_than_all = True
                for circle in circles:
                    distance = sqrt((circle[0] - i)**2 + (circle[1] - j)**2)
                    if distance < distance_threshold:
                        if circle[2] < current_value:
                            circle = (i,j,current_value)
                        larger_than_all = False
                if larger_than_all:
                    circles.append((i,j,current_value))
    return circles

def generate_circle_map(accumulator_arr: np.ndarray, radius: Tuple[int,int], circles) -> np.ndarray:
    radius_avg = int(np.average([radius[0],radius[1]]))
    height, width = accumulator_arr.shape
    circle_map = np.zeros(accumulator_arr.shape)
    for circle in circles:
        circle_map = cv2.circle(circle_map, (circle[1],circle[0]), radius_avg, color = [255,255,255], thickness = 2)
    return circle_map

def create_final_result(input_image: np.ndarray, circle_map: np.ndarray) -> np.ndarray:
    result = input_image.copy()
    height, width = circle_map.shape
    for i in range(0,height):
        for j in range(0,width):
            if circle_map[i,j] > 0:
                result[i,j] = 255
    return result

def hough_transform(input_data: Tuple[str,int,int,float, int]):
    image_name, rmin, rmax, threshold, distance_threshold = input_data
    input_asgrayscale = open_image_as_grayscale(image_name= image_name)
    binary_image = preprocess_image(grayscale_img= input_asgrayscale)
    #plt.imshow(binary_image,cmap='gray')
    #plt.show()
    
    points = generate_predifined_circlepoints(radius_min= rmin, radius_max= rmax)
    acc_arr = generate_accumulation_matrix(binary_image, points= points)
    #plt.imshow(acc_arr, cmap='gray')
    cv2.imwrite(image_name + '_accumulator_result.png', acc_arr.astype(np.uint8))
    #plt.show()
    
    acc_arr = remove_noise_from_accumulator(accumulator_arr= acc_arr, threshold= threshold)
    #plt.imshow(acc_arr, cmap='gray')
    #plt.show()
    
    circles = generate_circles(accumulator_arr = acc_arr, distance_threshold = distance_threshold)
    print(len(circles))
    
    circle_map = generate_circle_map(accumulator_arr= acc_arr, radius=(rmin,rmax), circles = circles)
    #plt.imshow(circle_map, cmap='gray')
    #plt.show()
    
    final_res = create_final_result(input_image= input_asgrayscale,circle_map= circle_map)
    #plt.imshow(final_res, cmap='gray')
    cv2.imwrite(image_name + '_final_result.png', final_res.astype(np.uint8))
    #plt.show()

input1 = ('circles.png',12,15,0.95,25)
#input2 = ('blood.png', 8, 12, 0.55,20)
#input3 = ('cable.png', 18, 25, 0.46,35)
#input4 = ('cells.png', 8, 18, 0.45,25)
hough_transform(input_data= input1)



