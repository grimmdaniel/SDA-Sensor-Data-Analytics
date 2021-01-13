import numpy as np
import cv2
import datetime

class ImageKernel:
    
    def __init__(self, image_filter: np.ndarray, constant: np.float32):
        self.image_filter = image_filter
        self.constant = constant
        
    @property
    def kernel_size(self):
        return self.image_filter.shape[0]
    
    @property
    def threshold_size(self):
        return self.kernel_size // 2

def open_image_as_grayscale(image_name: str) -> np.ndarray:
    img = cv2.imread(image_name)
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return grayscale_img.astype(np.int32)

def apply_filter_on_imagewindow(window: np.ndarray, imgfilter: ImageKernel) -> np.int32:
    sum_of_multiplication: np.int32 = 0
    for i in range(imgfilter.kernel_size):
        for j in range(imgfilter.kernel_size):
            sum_of_multiplication += (window[i][j] * imgfilter.image_filter[i][j])
    return sum_of_multiplication * imgfilter.constant

def create_empty_image(height: int, width: int) -> np.ndarray:
    empty_image = np.zeros(shape=(height,width), dtype = np.int32)
    return empty_image

def convolve_image(image: np.ndarray, image_kernel: ImageKernel) -> np.ndarray:
    
    #Filling borders of the new image with the mean value of the original image
    def calculate_image_mean(image: np.ndarray) -> np.int32:
        summed_pixels = np.int32(image.sum() / image.size)
        return summed_pixels

    image_height, image_width = image.shape
    threshold = image_kernel.threshold_size
    empty_image = create_empty_image(image_height,image_width)
    image_mean = calculate_image_mean(image)
    empty_image += image_mean
    for i in range(threshold,image_height-threshold):
        for j in range(threshold,image_width-threshold):
            window = (image[i-threshold:i+threshold+1, j-threshold:j+threshold+1])
            filtered_value = apply_filter_on_imagewindow(window, image_kernel)
            empty_image[i][j] = filtered_value 
    return empty_image

def process_image_with_filter(grayscale_img: np.ndarray, image_kernel_x: ImageKernel, image_kernel_y: ImageKernel):
    convolve_x = convolve_image(image= grayscale_img,
                                         image_kernel= image_kernel_x)
    convolve_y = convolve_image(image= grayscale_img,
                                         image_kernel= image_kernel_y)
    
    #Calculating magnitude np.hypot is equal to sqrt(x^2 + y^2)
    magnitude = np.hypot(convolve_x, convolve_y)
    
    #Normalizing values to be between 0 and 255 (prewitt masks with 1/3 multiplier give different results)
    magnitude = magnitude / magnitude.max() * 255
    
    #Calculating orientation matrix
    orientation = np.arctan2(convolve_y, convolve_x)
    return (magnitude, orientation)
            
def non_maxima_suppression(magnitude: np.ndarray, orientation: np.ndarray):
    image_height, image_width = magnitude.shape
    result = np.zeros((image_height, image_width), dtype=np.int32)
    orientation_angle = orientation * 180. / np.pi
    orientation_angle[orientation_angle < 0] += 180
    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):
            try:
                MA = 255
                MB = 255
                
                #Getting line direction
                #angle 0
                MC = orientation_angle[i,j]
                if (0 <= MC < 22.5) or (157.5 <= MC <= 180):
                    MA = magnitude[i, j+1]
                    MB = magnitude[i, j-1]
                #angle 45
                elif (22.5 <= MC < 67.5):
                    MA = magnitude[i+1, j-1]
                    MB = magnitude[i-1, j+1]
                #angle 90
                elif (67.5 <= MC < 112.5):
                    MA = magnitude[i+1, j]
                    MB = magnitude[i-1, j]
                #angle 135
                elif (112.5 <= MC < 157.5):
                    MA = magnitude[i-1, j-1]
                    MB = magnitude[i+1, j+1]
                
                current_magnitude = magnitude[i,j]
                if (current_magnitude >= MA) and (current_magnitude >= MB):
                    result[i,j] = current_magnitude
                else:
                    result[i,j] = 0

            except IndexError:
                pass
    return result
    
def main():
    t1 = datetime.datetime.utcnow()
    
    #Creating prewitt x and y masks
    gradient_kernel_prewitt_x: ImageKernel = ImageKernel(image_filter= np.array([[-1, 0, 1],
                                                                               [-1, 0, 1],
                                                                               [-1, 0, 1]],
                                                                              dtype = np.int32),
                                            constant= 1)
    gradient_kernel_prewitt_y: ImageKernel = ImageKernel(image_filter= np.array([[1, 1, 1],
                                                                               [0, 0, 0],
                                                                               [-1, -1, -1]],
                                                                              dtype = np.int32),
                                            constant= 1)
    
    #Opening image in grayscale
    image_name: str = 'circlegrey.png'
    test_image = open_image_as_grayscale(image_name= image_name)
    
    #Getting magnitude and orientation from input image
    magnitude, orientation = process_image_with_filter(grayscale_img= test_image,
                              image_kernel_x= gradient_kernel_prewitt_x,
                              image_kernel_y= gradient_kernel_prewitt_y)
    
    #Performing NMS with magnitude and orientation matrices
    suppressed_image = (non_maxima_suppression(magnitude= magnitude, orientation= orientation)).astype(np.int32)
    
    t2 = datetime.datetime.utcnow()
    #Measuring execution time
    print('Elapsed time: ' + str(t2-t1))
    
    #Writing result images to working directory
    cv2.imwrite(image_name + '_gradient_magnitude.png', magnitude.round().astype(np.uint8)) 
    cv2.imwrite(image_name + '_nms.png',suppressed_image.astype(np.uint8))
    
    #cv2.imshow('Original image', test_image.astype(np.uint8))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
main()