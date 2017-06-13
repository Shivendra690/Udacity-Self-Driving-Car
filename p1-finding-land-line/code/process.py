#importing some useful packages 
from __future__ import print_function
import cv2
from utils import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from moviepy.editor import VideoFileClip
import os
import argparse

# Pipeline that will draw lane lines on IMG
def process_pipeline(img):
    # Convert image to grayscale
    img_gray = grayscale(img)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 7
    blur_gray = gaussian_blur(img_gray, kernel_size)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    # Define our parameters for image mask and apply 
    imshape = img.shape
    vertices = np.array([[(0,imshape[0]),(450, 320), (490, 320), (imshape[1],imshape[0])]], dtype=np.int32)
    
    masked_edges = region_of_interest(edges, vertices)
    

    # Define the Hough transform parameters and apply
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 5     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 20 # minimum number of pixels making up a line
    max_line_gap = 5    # maximum gap in pixels between connectable line segments
    
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
    
    # Weigh the image and return
    return weighted_img(line_image, img)

def process_image(image):
    # The output returned is a color image (3 channel) for processing video below
    result = process_pipeline(image)
    return result

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description="This script finds land lines in a video and output a processed video with red marking over the land lines found.")
    parser.add_argument("video", metavar='v', type=str, help="The video that you want to find land lines in. Format: ____.mp4")
    args = parser.parse_args()

    
    # Prepare output directory
    output_directory = "processed_output"
    if output_directory not in os.listdir('.'):
        os.mkdir(output_directory)
    name = args.video.split("/")[-1]
    path = "{0}/{1}".format(output_directory, name)
    if path in os.listdir(output_directory):
        os.remove(path)

    # Process and save video
    print("Process Video...")
    clip = VideoFileClip(args.video)
    clip = clip.fl_image(process_image)
    print("Saving Processed video to", path)
    clip.write_videofile(path, audio=False)
    print("Video Saved...")

