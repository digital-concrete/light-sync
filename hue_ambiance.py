# coding: utf-8
"""
Script for synchronizing Philips Hue lights with computer display colors.

Based on:

http://python-mss.readthedocs.io/examples.html

https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html

https://www.developers.meethue.com/

https://github.com/studioimaginaire/phue

https://github.com/benknight/hue-python-rgb-converter

Hue Personal Wireless Lighting is a trademark owned by Philips Electronics
See www.meethue.com for more information.
I am in no way affiliated with the Philips organization.

Published under the MIT license - See LICENSE file for more details.

Copyright (c) 2018 DIGITAL CONCRETE JUNGLE / MIT License.
"""

from __future__ import division

import math
import time
import threading
import numpy
import cv2
import mss

from phue import Bridge
from convertor_lib import Converter

# SETUP VARIABLES (tweak them for preffered effect)

# Your Philips Hue Bridge IP
BRIDGE_IP = '192.168.1.22'

# Your Philips HUE lights that will be updated by this script
MY_LIGHT_NAMES = ['Light1', 'Light2']

# Max number of Hue update requests per second. Used to prevent Hue Bridge bottleneck
HUE_MAX_REQUESTS_PER_SECOND = 3

# Use to skip similar frames in a row
FRAME_MATCH_SENSITIVITY = 0.02

# Resulted colors from previous and current frame are compared channel by channel
# If there is no difference bigger than COLOR_SKIP_SENSITIVITY between any of the channels
# then frame is skipped
COLOR_SKIP_SENSITIVITY = 20

# If color channel values are below or above these values they are considered to be dark or bright
# When most of the screen is dark or bright then the next available color is sent to Philips HUE
CHANNELS_MIN_THRESHOLD = 60
CHANNELS_MAX_THRESHOLD = 190

# MIN NON ZERO COUNT
MIN_NON_ZERO_COUNT = 100

# Min color spread threshold
COLOR_SPREAD_THRESHOLD = 0.005

# Number of clusters computed by the OpenCV K-MEANS algorithm
NUMBER_OF_K_MEANS_CLUSTERS = 6

# Captured screen can have a very large amount of data which takes longer time to process
# by the K Means algorithm.
# Image will be scaled to a much smaller size resulting in real time updating of the lights
INPUT_IMAGE_REDUCED_SIZE = 200

# Transition time
TRANSITION_TIME = 1

# Starting brightness
STARTING_BRIGHTNESS = 80

# Frame Color Definition


class FrameColor(object):
    """__init__() functions as the class constructor"""

    def __init__(self, color=None, index=None, color_count=0):
        self.color = color
        self.index = index
        self.color_count = color_count
        self.is_dark = False
        self.id_bright = False
        self.calculate_light_dark_channels()

    def calculate_light_dark_channels(self):
        """Calculates whether color is bright or dark"""
        bright_channels_count = 0
        dark_channels_count = 0
        for channel in range(0, 3):
            if self.color[channel] > CHANNELS_MAX_THRESHOLD:
                bright_channels_count += 1

            if self.color[channel] < CHANNELS_MIN_THRESHOLD:
                dark_channels_count += 1

        if bright_channels_count == 3:
            self.is_bright = True

        if dark_channels_count == 3:
            self.is_dark = True

    def get_hue_color(self):
        """Return the color in Philips HUE XY format"""
        return COLOR_CONVERTER.rgb_to_xy(
            self.color[2], self.color[1], self.color[0])


def shrink_image(input_img):
    "Reduce image size to increase computation speed"
    height, width = input_img.shape[:2]
    max_height = INPUT_IMAGE_REDUCED_SIZE
    max_witdh = INPUT_IMAGE_REDUCED_SIZE

    if max_height < height or max_witdh < width:
        # Get scaling factor
        scaling_factor = max_height / float(height)
        if max_witdh/float(width) < scaling_factor:
            scaling_factor = max_witdh / float(width)
        # Resize image. You can use INTER_AREA if you have a performant computer
        input_img = cv2.resize(input_img, None, fx=scaling_factor,
                               fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
    return input_img


def calculate_hue_color(input_img):
    "Calculates the color to be sent to HUE"
    # COMPUTE K MEANS
    k_means_input_img = input_img.reshape((-1, 4))

    # Convert to np.float32
    k_means_input_img = numpy.float32(k_means_input_img)

    # Define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = NUMBER_OF_K_MEANS_CLUSTERS
    ret, label, center = cv2.kmeans(
        k_means_input_img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = numpy.uint8(center)

    # COMPUTE MOST PREVALENT CLUSTER

    # Calculate the prevalence for each one of the resulted colors
    label_counts = []
    for j in range(NUMBER_OF_K_MEANS_CLUSTERS):
        label_counts.append(0)
    for j in range(0, label.size):
        label_counts[label[j][0]] += 1

    # Init and populate Array of FrameColor objects for further calculations/decisions
    frame_colors = []

    for j in range(NUMBER_OF_K_MEANS_CLUSTERS):
        frame_color = FrameColor(center[j], j, label_counts[j])
        frame_colors.append(frame_color)

    # Sort by prevalence
    frame_colors.sort(key=lambda x: x.color_count, reverse=True)

    # Calculate color to be sent to Hue
    result_color = frame_colors[0]

    if frame_colors[0].is_bright:
        for j in range(1, NUMBER_OF_K_MEANS_CLUSTERS):
            if (not frame_colors[j].is_bright
                    and not frame_colors[j].is_dark
                    and frame_colors[j].color_count / label.size
                    > COLOR_SPREAD_THRESHOLD):
                result_color = frame_colors[j]
                break

    if frame_colors[0].is_dark:
        for j in range(1, NUMBER_OF_K_MEANS_CLUSTERS):
            if not frame_colors[j].is_dark:
                result_color = frame_colors[j]
                break

    return result_color


# CAN UPDATE HUE FLAG
CAN_UPDATE_HUE = True


def clear_update_flag():
    """Clears the hue update request lock"""
    # can_update_hue
    global CAN_UPDATE_HUE
    CAN_UPDATE_HUE = True


# Init convertor lib used to format color for HUE
COLOR_CONVERTER = Converter()

# Your bridge IP
BRIDGE = Bridge(BRIDGE_IP)


def main():
    "main routine"
    # Variables
    go_dark = False
    lights_are_on = True
    prev_color = None
    prev_frame = None
    request_timeout = 1/HUE_MAX_REQUESTS_PER_SECOND

    global CAN_UPDATE_HUE

    # If the app is not registered and the button is not pressed,
    # press the button and call connect() (this only needs to be run a single time)
    BRIDGE.connect()

    print 'Connected to Hue Bridge with address {0}'.format(BRIDGE_IP)

    light_names = BRIDGE.get_light_objects('name')

    # First make sure lights are on
    for hue_light in MY_LIGHT_NAMES:
        light_names[hue_light].on = True
        light_names[hue_light].transitiontime = TRANSITION_TIME
        light_names[hue_light].brightness = STARTING_BRIGHTNESS

    with mss.mss() as sct:
        # Part of the screen to capture (use if you want to create a multiple color effect)
        # monitor = {'top': 40, 'left': 0, 'width': 800, 'height': 640}
        monitor = sct.monitors[1]

        while 'Screen capturing':
            last_time = time.time()

            # Get raw pixels from the screen, save it to a Numpy array
            img = numpy.array(sct.grab(monitor))

            current_frame = shrink_image(img)

            # Compare Frame with Prev Frame
            # Skip if similar
            if prev_frame is not None and lights_are_on:
                comparison_result = cv2.matchTemplate(
                    current_frame, prev_frame, 1)
                if comparison_result[0][0] < FRAME_MATCH_SENSITIVITY:
                    continue

            # Apply dark color threshold and compute mask
            gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(
                gray_frame, CHANNELS_MIN_THRESHOLD, 255, cv2.THRESH_BINARY)

            # Apply mask to frame
            masked_frame = cv2.bitwise_and(
                current_frame, current_frame, mask=mask)

            # Check the non zero pixels. If their count is too low turn the lights off
            gray_image_output = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
            nz_count = cv2.countNonZero(gray_image_output)
            if nz_count < MIN_NON_ZERO_COUNT:
                go_dark = True
            else:

                # If False go through all routines
                go_dark = False

                # Calculate relevant color for this frame
                result_color = calculate_hue_color(masked_frame)

                # Compare Current Calculated Color with previous Color
                # Skip frame if result color is almost identical
                if prev_color is not None and not go_dark and lights_are_on:
                    skip_frame = True

                    for j in range(0, 3):
                        ch_diff = math.fabs(
                            int(prev_color.color[j]) - int(result_color.color[j]))
                        if ch_diff > COLOR_SKIP_SENSITIVITY:
                            skip_frame = False
                            break
                    if skip_frame:
                        continue

            # Send color to Hue if update flag is clear
            if CAN_UPDATE_HUE:

                # Compare only with last used frame/color
                if go_dark:
                    prev_color = None
                else:
                    prev_color = result_color

                prev_frame = current_frame

                # Timer that limits the requests to hue bridge in order to prevent bottlenecks
                CAN_UPDATE_HUE = False
                update_timer = threading.Timer(
                    request_timeout, clear_update_flag)
                update_timer.start()

                # SWITCH ON/OFF Logic
                switch_lights = False
                if go_dark and lights_are_on:
                    lights_are_on = False
                    switch_lights = True
                    print 'Switch off for now'
                if not go_dark and not lights_are_on:
                    lights_are_on = True
                    switch_lights = True
                    print 'Switch back on'
                else:
                    print 'Updating with RGB: [{0}, {1}, {2}]'.format(
                        result_color.color[0], result_color.color[1], result_color.color[2])

                for hue_light in MY_LIGHT_NAMES:

                    # COMMENT OR REMOVE THIS BLOCK IF YOU DON'T WANT TO SWITCH ON/OFF YOUR LIGHTS
                    if switch_lights:
                        light_names[hue_light].on = lights_are_on

                    if lights_are_on:
                        light_names[hue_light].xy = result_color.get_hue_color()

                print 'fps: {0}'.format(1 / (time.time()-last_time))


main()
