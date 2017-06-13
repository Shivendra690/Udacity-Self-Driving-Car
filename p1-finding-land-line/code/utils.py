import cv2
import math
import numpy as np

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=15):
    """
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    def remove_outliers(x, outlierConstant):
        a = np.array(x)
        upper_quartile = np.percentile(a, 75)
        lower_quartile = np.percentile(a, 25)
        IQR = (upper_quartile - lower_quartile) * outlierConstant
        quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
        resultList = []
        for y in a.tolist():
            if y > quartileSet[0] and y < quartileSet[1]:
                resultList.append(y)
        return resultList

    # Average the positions of lines in format: x1, y1, x2, y2
    def get_mean_coordinates(lines):
        x1s, x2s, y1s, y2s = list(), list(), list(), list()
        slopes = list()
        for line in lines:
            for x1, y1, x2, y2 in line:
                x1s.append(x1)
                x2s.append(x2)
                y1s.append(y1)
                y2s.append(y2)
                if x1 > x2:
                    x1, x2 = x2, x1
                    y1, y2 = y2, y1
                slope = (y2 - y1) / (x2 - x1)
                slopes.append(slope)
        upper_quartile = np.percentile(slopes, 75)
        lower_quartile = np.percentile(slopes, 25)
        IQR = (upper_quartile - lower_quartile) * 1.5
        bounds = (lower_quartile - IQR, upper_quartile + IQR)
        remove_index_list = []
        for i, s in enumerate(slopes):
            if s < bounds[0] or s > bounds[1]:
                remove_index_list.append(i)
        x1s = [x for i, x in enumerate(x1s) if i not in remove_index_list]
        y1s = [x for i, x in enumerate(y1s) if i not in remove_index_list]
        x2s = [x for i, x in enumerate(x2s) if i not in remove_index_list]
        y2s = [x for i, x in enumerate(y2s) if i not in remove_index_list]
        return np.mean(x1s), np.mean(y1s), np.mean(x2s), np.mean(y2s)
    
    def extrapolate(x1, y1, x2, y2, left=True):
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        elif x1 == x2:
            if left:
                return x1, img.shape[0], x2, 320
            else:
                return x1, 320, x2, img.shape[0]
            
        def get_line(x1, y1, x2, y2):
            if x1 > x2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            slope = (y2 - y1) / (x2 - x1)
            def line(a, getX=True):
                # y - y1 = slope(x - x1)
                if getX:
                    return (a - y1 + slope * x1) / slope
                else:
                    return slope * (a - x1) + y1
            return line
        
        line = get_line(x1, y1, x2, y2)
        if left:
            x_limit = (0, 450)
            y_limit = (330, img.shape[0])
            x1_new = x_limit[0]
            y1_new = line(x1_new, getX=False)
            if y1_new > y_limit[1]:
                y1_new = y_limit[1]
                x1_new = line(y1_new, getX=True)
            x2_new = x_limit[1]
            y2_new = line(x2_new, getX=False)
            if y2_new < y_limit[0]:
                y2_new = y_limit[0]
                x2_new = line(y2_new, getX=True)
        else:
            x_limit = (490, img.shape[1])
            y_limit = (330, img.shape[0])
            x1_new = x_limit[0]
            y1_new = line(x1_new, getX=False)
            if y1_new < y_limit[0]:
                y1_new = y_limit[0]
                x1_new = line(y1_new, getX=True)
            x2_new = x_limit[1]
            y2_new = line(x2_new, getX=False)
            if y2_new > y_limit[1]:
                y2_new = y_limit[1]
                x2_new = line(y2_new, getX=True)
        return int(x1_new), int(y1_new), int(x2_new), int(y2_new)
    
    
    # Separate lines to left and right
    left_lines = list()
    right_lines = list()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 >= x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
        slope = (y2 - y1) / (x2 - x1)
        if slope <= 0:
            left_lines.append(line)
        else:
            right_lines.append(line)
    
    if len(left_lines) >= 1:
        x1_left, y1_left, x2_left, y2_left = get_mean_coordinates(left_lines)
        x1_left, y1_left, x2_left, y2_left = extrapolate(x1_left, y1_left, x2_left, y2_left, left=True)
        cv2.line(img, (x1_left, y1_left), (x2_left, y2_left), color, thickness)
    
    if len(right_lines) >= 1:
        x1_right, y1_right, x2_right, y2_right = get_mean_coordinates(right_lines)
        x1_right, y1_right, x2_right, y2_right = extrapolate(x1_right, y1_right, x2_right, y2_right, left=False)
        cv2.line(img, (x1_right, y1_right), (x2_right, y2_right), color, thickness)

    

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

