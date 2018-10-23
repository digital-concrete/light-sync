# coding: utf-8
"""
Frame color related lib
Helper methods and
Most relevand color algorithm
"""

from __future__ import print_function
import numpy
import cv2
import math

from convertor_lib import Converter

# Frame Color Definition
class FrameColor(object):
    """__init__() functions as the class constructor"""

    def __init__(self, color, index, color_count,
                 min_threshold, max_threshold, color_converter):
        self.color = color
        self.index = index
        self.color_count = color_count
        self.is_dark = False
        self.is_bright = False
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.calculate_light_dark_channels()
        self.color_converter = color_converter
        self.brightness = None
        self.go_dark = None
        self.diff_from_prev = None

    def calculate_light_dark_channels(self):
        """Calculates whether color is bright or dark"""
        bright_channels_count = 0
        dark_channels_count = 0
        for channel in range(0, 3):
            if self.color[channel] > self.max_threshold:
                bright_channels_count += 1

            if self.color[channel] < self.min_threshold:

                dark_channels_count += 1

        if bright_channels_count == 3:
            self.is_bright = True

        if dark_channels_count == 3:
            self.is_dark = True

    def get_hue_color(self):
        """Return the color in Philips HUE XY format"""
        return self.color_converter.rgb_to_xy(
            self.color[2], self.color[1], self.color[0])

class FrameColorLib:
    def __init__(self):
        "init"
        self.color_converter = Converter()  

    def shrink_image(self, input_img, input_image_reduced_size):
        "Reduce image size to increase computation speed"
        height, width = input_img.shape[:2]
        max_height = input_image_reduced_size
        max_witdh = input_image_reduced_size

        if max_height < height or max_witdh < width:
            # Get scaling factor
            scaling_factor = max_height / float(height)
            if max_witdh/float(width) < scaling_factor:
                scaling_factor = max_witdh / float(width)
            # Resize image. You can use INTER_AREA if you have a performant computer
            input_img = cv2.resize(input_img, None, fx=scaling_factor,
                                fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
        return input_img

    def apply_frame_mask(self, current_frame, channels_min_threshold):
        "Apply dark color threshold and compute mask"
        gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(
            gray_frame, channels_min_threshold, 255, cv2.THRESH_BINARY)

        # Apply mask to frame
        masked_frame = cv2.bitwise_and(
            current_frame, current_frame, mask=mask)
        
        return masked_frame

    def calculate_frame_brightness(self, frame, dim_brightness, starting_brightness,
                                    min_non_zero_count, max_non_zero_count):
        "Calculates frame brightness"

        # Actual non zero thresholds in pixels
        min_non_zero_count_pixels = frame.size * min_non_zero_count / 100
        max_non_zero_count_pixels = frame.size * max_non_zero_count / 100

        # Check the non zero pixels. If their count is too low turn the lights off
        gray_image_output = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        nz_count = cv2.countNonZero(gray_image_output)
        if nz_count < min_non_zero_count_pixels:
            current_brightness = dim_brightness - 1
        else:
            # If False go through all routines
            if nz_count > max_non_zero_count_pixels:
                current_brightness = starting_brightness
            else:
                current_brightness = (nz_count - min_non_zero_count_pixels) * (
                    starting_brightness - dim_brightness) / (
                        max_non_zero_count_pixels - min_non_zero_count_pixels) + (
                            dim_brightness)
                current_brightness = int(current_brightness)
        
        return current_brightness

    def frame_colors_are_similar(self, first_color, second_color,
                                    color_skip_sensitivity,
                                    brightness_skip_sensitivity):
        "checks if 2 frame colors are similar"
        result = False
        if first_color is not None and\
                second_color is not None:
                if(first_color.go_dark == True and second_color.go_dark == True):
                    return True

                if abs(first_color.brightness - second_color.brightness) < brightness_skip_sensitivity:
                    for j in range(0, 3):
                        ch_diff = math.fabs(
                            int(first_color.color[j]) - int(second_color.color[j]))
                        if ch_diff < color_skip_sensitivity:
                            result = True
                            break
        return result

    def calculate_hue_color(self, input_img, k_means,
                            color_spread_threshold,
                            channels_min_threshold,
                            channels_max_threshold):
        "Calculates the color to be sent to HUE"
        # COMPUTE K MEANS
        k_means_input_img = input_img.reshape((-1, 4))

        # Convert to np.float32
        k_means_input_img = numpy.float32(k_means_input_img)

        # Define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = k_means
        ret, label, center = cv2.kmeans(
            k_means_input_img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = numpy.uint8(center)

        # COMPUTE MOST PREVALENT CLUSTER

        # Calculate the prevalence for each one of the resulted colors
        label_counts = []
        for j in range(k_means):
            label_counts.append(0)
        for j in range(0, label.size):
            label_counts[label[j][0]] += 1

        # Init and populate Array of FrameColor objects for further calculations/decisions
        frame_colors = []

        for j in range(k_means):
            frame_color = FrameColor(center[j], j, label_counts[j],
                                    channels_min_threshold, channels_max_threshold, self.color_converter)
            frame_colors.append(frame_color)

        # Sort by prevalence
        frame_colors.sort(key=lambda x: x.color_count, reverse=True)

        # Calculate color to be sent to Hue
        result_color = frame_colors[0]

        if frame_colors[0].is_bright:
            for j in range(1, k_means):
                if (not frame_colors[j].is_bright
                        and not frame_colors[j].is_dark
                        and (frame_colors[j].color_count / label.size) * 100
                        > color_spread_threshold):
                    result_color = frame_colors[j]
                    break

        if frame_colors[0].is_dark:
            for j in range(1, k_means):
                if not frame_colors[j].is_dark:
                    result_color = frame_colors[j]
                    break

        return result_color

   