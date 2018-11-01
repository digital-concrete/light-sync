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
from __future__ import print_function

import sys
import os
import argparse
import math
import time
import threading
import numpy
import cv2
import mss
import requests
import json

from phue_lib import Bridge, PhueRegistrationException
from discovery_lib import DiscoveryLib
from frame_color_lib import FrameColorLib

# SETUP VARIABLES (tweak them for preffered effect)

# Your Philips Hue Bridge IP
BRIDGE_IP = '192.168.1.22'

# Part of screen to capture (useful if run in multiple instance for custom effects like stereo)
# full, left, right, side-left side-right
SCREEN_PART_TO_CAPTURE = "full"

# Your Philips HUE lights that will be updated by this script
MY_LIGHT_NAMES = ['Light1','Light2']

# IDS of Your Philips HUE lights that will be updated by this script
MY_LIGHT_IDS = []

# Dim lights instead of turn off
DIM_LIGHTS_INSTEAD_OF_TURN_OFF = False

# MAX brightness
STARTING_BRIGHTNESS = 110

# Dim brightness
DIM_BRIGHTNESS = 3

# Default color to be used when RGB are all 0
DEFAULT_DARK_COLOR = [64, 75, 78]

# Skip frame if brightness difference is less than this
BRIGHTNESS_SKIP_SENSITIVITY = 10

# Transition type: 
# 0 = instant
# 1 = instant if frame diff is over threshold and smooth if frames are similar
# 2 = smooth
TRANSITION_TYPE = 0

# Transition time for smooth transitions
TRANSITION_TIME = 1

# Max number of Hue update requests per second. Used to prevent Hue Bridge bottleneck
HUE_MAX_REQUESTS_PER_SECOND = 10

# Use to skip similar frames in a row
FRAME_MATCH_SENSITIVITY = 0.008

# BETA version!!!
# Use to transition smoothly when frames are less than this value and
# greater than FRAME_MATCH_SENSITIVITY 
FRAME_MATCH_SMOOTH_TRANSITION_SENSITIVITY = 0.1

# Resulted colors from previous and current frame are compared channel by channel
# If there is no difference bigger than COLOR_SKIP_SENSITIVITY between any of the channels
# then frame is skipped
COLOR_SKIP_SENSITIVITY = 10

# If color channel values are below or above these values they are considered to be dark or bright
# When most of the screen is dark or bright then the next available color is considered
CHANNELS_MIN_THRESHOLD = 50
CHANNELS_MAX_THRESHOLD = 190

# MIN NON ZERO COUNT
MIN_NON_ZERO_COUNT = 0.2

# TOP NON ZERO COUNT THRESHOLD
MAX_NON_ZERO_COUNT = 20

# Min color spread threshold
COLOR_SPREAD_THRESHOLD = 0.4

# Number of clusters computed by the OpenCV K-MEANS algorithm
NUMBER_OF_K_MEANS_CLUSTERS = 6

# Captured screen can have a very large amount of data which takes longer time to process
# by the K Means algorithm.
# Image will be scaled to a much smaller size resulting in real time updating of the lights
INPUT_IMAGE_REDUCED_SIZE = 100

# PHUE config file name
PHUE_CONFIG_FILE = "phue_config"

# BETA version!!!
# Auto adjust performance
AUTO_ADJUST_PERFORMANCE = True

# Auto adjust performance after number of low fps in a row
AUTO_ADJUST_PERFORMANCE_NUMBER_OF_LOW_FPS_IN_A_ROW = 2

# BETA version!!!
# Flicker prevent
FLICKER_PREVENT = True

# Number of frames in buffer
FLICKER_PREVENT_BUFFER_SIZE = 20

# Number of frames to check for color flicker
COLOR_FLICKER_PREVENT_REQUIRED_INPUT_FRAMES_COUNT = 4

# How many frames satisfy the conditions in order to detect color flicker
COLOR_FLIKCER_DETECTION_THRESHOLD = 2

# Number of frames to check for on/off flicker
ON_OFF_FLICKER_PREVENT_REQUIRED_INPUT_FRAMES_COUNT = 8

# How many frames satisfy the conditions in order to detect on/off flicker
ON_OFF_FLIKCER_DETECTION_THRESHOLD = 2

# Number of frames with correction until reset
FLICKER_CORRECTION_FRAME_COUNT_UNTIL_RESET = 30

# Flicker reset sensitivity
# If tho consecutive frames have a greater match diff,
# then reset the adjustments
FLICKER_CORRECTION_RESET_SENSITIVITY = 0.3

# Color Flicker sensitivity
# Color flicker is detected if frame match is lower than this
# and result colors are different
COLOR_FLICKER_FRAME_MATCH_THRESHOLD = 0.2

# ON OFF FLICKER CORRECTION
ON_OFF_FLICKER_MIN_NON_ZERO_COUNT_CORRECTION_VALUE = 0.1

# ON OFF FLICKER CORRECTION
ON_OFF_FLICKER_MIN_THRESHOLD_FLICKER_CORRECTION_VALUE = 0.2

# GLOBALS

# CAN UPDATE HUE FLAG
CAN_UPDATE_HUE = True

# Init convertor lib used to format color for HUE
DISCOVERY_LIB = DiscoveryLib()
FRAME_COLOR_LIB = FrameColorLib()

def clear_update_flag():
    """Clears the hue update request lock"""
    # can_update_hue
    global CAN_UPDATE_HUE
    CAN_UPDATE_HUE = True

def usage(parser):
    """Help"""
    parser.print_help()
    
    print ("Example:")
    print ("\t" + sys.argv[0] +
           " --bridgeip 192.168.1.23 --lights Light1,Light2")
    sys.exit()

def main(argv):
    "main routine"

    # Variable Defaults
    bridge_ip = BRIDGE_IP
    user = None
    screen_part_to_capture = SCREEN_PART_TO_CAPTURE
    my_light_names = MY_LIGHT_NAMES
    my_light_ids = MY_LIGHT_IDS
    dim_brightness = DIM_BRIGHTNESS
    starting_brightness = STARTING_BRIGHTNESS
    dim_lights_instead_of_turn_off = DIM_LIGHTS_INSTEAD_OF_TURN_OFF
    transition_time = TRANSITION_TIME
    transition_type = TRANSITION_TYPE
    frame_transition_sensitivity = FRAME_MATCH_SMOOTH_TRANSITION_SENSITIVITY
    frame_match_sensitivity = FRAME_MATCH_SENSITIVITY
    color_skip_sensitivity = COLOR_SKIP_SENSITIVITY
    brightness_skip_sensitivity = BRIGHTNESS_SKIP_SENSITIVITY
    channels_min_threshold = CHANNELS_MIN_THRESHOLD
    channels_max_threshold = CHANNELS_MAX_THRESHOLD
    min_non_zero_count = MIN_NON_ZERO_COUNT
    max_non_zero_count = MAX_NON_ZERO_COUNT
    color_spread_threshold = COLOR_SPREAD_THRESHOLD
    number_of_k_means_clusters = NUMBER_OF_K_MEANS_CLUSTERS
    input_image_reduced_size = INPUT_IMAGE_REDUCED_SIZE
    hue_max_requests_per_second = HUE_MAX_REQUESTS_PER_SECOND

    # Auto adjust performance
    auto_adjust_performance = AUTO_ADJUST_PERFORMANCE
    adjust_counter_limit = AUTO_ADJUST_PERFORMANCE_NUMBER_OF_LOW_FPS_IN_A_ROW

    # FLICKER related
    flicker_prevent = FLICKER_PREVENT
    result_buffer_size = FLICKER_PREVENT_BUFFER_SIZE
    flicker_correction_frame_count_until_reset = FLICKER_CORRECTION_FRAME_COUNT_UNTIL_RESET
    flicker_correction_reset_sensitivity = FLICKER_CORRECTION_RESET_SENSITIVITY
    color_flicker_frame_match_threshold = COLOR_FLICKER_FRAME_MATCH_THRESHOLD
    color_flicker_prevent_required_input_frames_count = COLOR_FLICKER_PREVENT_REQUIRED_INPUT_FRAMES_COUNT
    on_off_flicker_prevent_required_input_frames_count = ON_OFF_FLICKER_PREVENT_REQUIRED_INPUT_FRAMES_COUNT
    color_flicker_count_threshold = COLOR_FLIKCER_DETECTION_THRESHOLD
    on_off_flicker_count_threshold = ON_OFF_FLIKCER_DETECTION_THRESHOLD
    min_non_zero_count_on_off_flicker_correction_value = ON_OFF_FLICKER_MIN_NON_ZERO_COUNT_CORRECTION_VALUE
    channels_min_threshold_on_off_flicker_correction_value = ON_OFF_FLICKER_MIN_THRESHOLD_FLICKER_CORRECTION_VALUE

    # Arguments or defaults
    parser = argparse.ArgumentParser(description="Sync Hue Lights with computer display")
    parser.add_argument("-i", "--bridgeip", help="Your Philips Hue Bridge IP")
    parser.add_argument("-u", "--user", help="Your Philips Hue Bridge User")
    parser.add_argument("-p", "--screenpart", help="Part of the screen to capture: full, left, right, \
                                                side-left, side-right (default full)")
    parser.add_argument("-l", "--lightids", help="Your Philips HUE light Ids that will be updated, comma separated")
    parser.add_argument("-L", "--lights", help="Your Philips HUE light Names that will be updated, comma separated")
    parser.add_argument("-b", "--dimbrightness", help="Dim/MIN brightness [0-256] must be less than maxbrightness")
    parser.add_argument("-B", "--maxbrightness", help="MAX brightness [0-256]")
    parser.add_argument("-D", "--dimlightsinsteadofturnoff", help="Dim lights or Turn OFF on dark screens (default true - DIM)")
    parser.add_argument("-t", "--transitiontime", help="Transition time, default 1")
    parser.add_argument("-y", "--transitiontype", help="Transition type, default 1 (0 = instant, 1 = comfortable, 2 = smooth)")
    parser.add_argument("-v", "--frametransitionsensitivity", help="Smooth transition sensitivity for comfortable mode, [0-1] default 0.5.\
                                                             Frame difference between framematchsensitivity and frametransitionsensitivity\
                                                             will have a transition time. Otherwise transition will be instant")
    parser.add_argument("-s", "--framematchsensitivity", help="Use to skip similar frames in a row [0-1] default: 0.008")
    parser.add_argument("-S", "--colorskipsensitivity", help="Skip frame if color is similar [0-256], default 10")
    parser.add_argument("-g", "--brightnessskipsensitivity", help="Skip frame if brightness diff is less [0-256], default 10")
    parser.add_argument("-c", "--channelsminthreshold", help="Dark threshold [0-256], default 50")
    parser.add_argument("-C", "--channelsmaxthreshold", help="Bright threshold [0-25]6, default 190, > minthreshold")
    parser.add_argument("-m", "--minnzcount", help="Min non zero threshold [0-100], default 0.2")
    parser.add_argument("-M", "--maxnzcount", help="Top non zero threshold [1-100], default 20, > minthreshold")
    parser.add_argument("-d", "--colorspreadthreshold", help="Color spread threshold [0-100], default 0.005")
    parser.add_argument("-k", "--kmeansclusters", help="Number of clusters computed by the OpenCV K-MEANS algorithm")
    parser.add_argument("-z", "--shrinkframesize", help="Frame capture shrinked size in pixel for better performance (default 100)")
    parser.add_argument("-T", "--maxrequestspersecond", help="Max requests per second sent to bridge api (default 10)")
    parser.add_argument("-a", "--autodiscovery", help="Bridge auto discovery on LAN")
    
    # Auto adjust performance
    parser.add_argument("-A", "--autoadjustperformance", help="Auto adjust script performance (default True)")
    parser.add_argument("-Af", "--autoadjustperformancelowfpscount", help="Number of low fps in a row to trigger performance adjust (default 2)")

    # FLICKER args
    parser.add_argument("-f", "--flickercorrection", help="Enable flicker correction (default True)")
    parser.add_argument("-fb", "--flickerframebuffersize", help="Flicker prevent result buffer size (default 20)")
    parser.add_argument("-fr", "--flickerresetcounter", help="Flicker correction frame count until reset (default 30)")
    parser.add_argument("-fs", "--flickerresetsensitivity", help="Flicker correction reset sensitivity (default 0.3)")
    parser.add_argument("-fcs", "--colorflickersensitivity", help="Color flicker frame match threshold (default 0.2)")
    parser.add_argument("-fci", "--colorflickerinput", help="Color flicker detection required input frames count (default 4)")
    parser.add_argument("-foi", "--onoffflickerinput", help="On/Off flicker detection required input frames count (default 8)")
    parser.add_argument("-fcc", "--colorflickercount", help="Number of frames that match color flicker condition required to trigger correction (default 2)")
    parser.add_argument("-foc", "--onoffflickercount", help="Number of frames that match on/off flicker condition required to trigger correction (default 2)")
    parser.add_argument("-fonz", "--onoffflickerminnzcorrection", help="On/Off flicker min nz count correction (default 0.1)")
    parser.add_argument("-focm", "--onoffflickerchannelsmincorrection", help="On/Off flicker channels min threshold correction (default 0.2)")

    args = parser.parse_args()

    if args.bridgeip:
        bridge_ip = args.bridgeip
        print ("Set bridge ip: " + bridge_ip)
    if args.user:
        user = args.user
        print ("Set User: " + user)
    if args.screenpart:
        screen_part_to_capture = args.screenpart
        print ("Set screen part to capture: " + screen_part_to_capture)
    if args.autodiscovery:
        bridge_ip = DISCOVERY_LIB.getBridgeIP()
        print ("Discovered bridge ip: " + bridge_ip)
    if args.lightids:
        my_light_ids = args.lightids.split(",")
        print ("Set lights: " + ", ".join(my_light_ids))
    if args.lights:
        my_light_names = args.lights.split(",")
        print ("Set lights: " + ", ".join(my_light_names))
    if args.dimbrightness:
        try:
            dim_brightness = int(args.dimbrightness)
            print ("Set min/dim brightness: " + str(dim_brightness))
        except ValueError:
            print ("dimbrightness must be a number\n")
            usage(parser)
    if args.maxbrightness:
        try:
            starting_brightness = int(args.maxbrightness)
            print ("Set max brightness: " + str(dim_brightness))
        except ValueError:
            print ("maxbrightness must be a number\n")
            usage(parser)
    if args.dimlightsinsteadofturnoff:
        try:
            ua = str(args.dimlightsinsteadofturnoff).upper()
            if 'TRUE'.startswith(ua):
                dim_lights_instead_of_turn_off = True
            elif 'FALSE'.startswith(ua):
                dim_lights_instead_of_turn_off = False
            else:
                raise ValueError("dimlightsinsteadofturnoff must be a boolean\n")
            print ("Set dim lights or turn off: " + str(dim_lights_instead_of_turn_off))
        except ValueError:
            print ("dimlightsinsteadofturnoff must be a boolean\n")
            usage(parser)
    if args.transitiontime:
        try:
            transition_time = int(args.transitiontime)
            print ("Set transition time: " + str(transition_time))
        except ValueError:
            print ("transitiontime must be a number\n")
            usage(parser)
    if args.transitiontime:
        try:
            transition_type = int(args.transitiontime)
            print ("Set transition type: " + str(transition_type))
            print ("0 = instant, 1 = comfortable, 2 = smooth")
        except ValueError:
            print ("transitiontime must be a number\n")
            usage(parser)
    if args.frametransitionsensitivity:
        try:
            frame_transition_sensitivity = float(args.frametransitionsensitivity)
            print ("Set frame transistion sensitivity: " +
                    str(frame_transition_sensitivity))
        except ValueError:
            print ("frametransitionsensitivity must be a number\n")
            usage(parser)
    if args.framematchsensitivity:
        try:
            frame_match_sensitivity = float(args.framematchsensitivity)
            print ("Set frame match sensitivity: " +
                    str(frame_match_sensitivity))
        except ValueError:
            print ("framematchsensitivity must be a number\n")
            usage(parser)
    if args.colorskipsensitivity:
        try:
            color_skip_sensitivity = float(args.colorskipsensitivity)
            print ("Set color skip sensitivity: " +
                    str(color_skip_sensitivity))
        except ValueError:
            print ("colorskipsensitivity must be a number\n")
            usage(parser)
    if args.brightnessskipsensitivity:
        try:
            brightness_skip_sensitivity = float(args.brightnessskipsensitivity)
            print ("Set brightness skip sensitivity: " +
                    str(brightness_skip_sensitivity))
        except ValueError:
            print ("brightnessskipsensitivity must be a number\n")
            usage(parser)
    if args.channelsminthreshold:
        try:
            channels_min_threshold = int(args.channelsminthreshold)
            print ("Set channels min threshold: " +
                    str(channels_min_threshold))
        except ValueError:
            print ("channelsminthreshold must be a number\n")
            usage(parser)
    if args.channelsmaxthreshold:
        try:
            channels_max_threshold = int(args.channelsmaxthreshold)
            print ("Set channels max threshold: " +
                    str(channels_max_threshold))
        except ValueError:
            print ("channelsmaxthreshold must be a number\n")
            usage(parser)
    if args.minnzcount:
        try:
            min_non_zero_count = float(args.minnzcount)
            print ("Set min nz count: " + str(min_non_zero_count))
        except ValueError:
            print ("minnzcount must be a number\n")
            usage(parser)
    if args.maxnzcount:
        try:
            max_non_zero_count = float(args.maxnzcount)
            print ("Set max nz count: " + str(max_non_zero_count))
        except ValueError:
            print ("maxnzcount must be a number\n")
            usage(parser)
    if args.colorspreadthreshold:
        try:
            color_spread_threshold = float(args.colorspreadthreshold)
            print ("Set color spread: " +
                    str(color_spread_threshold))
        except ValueError:
            print ("colorspreadthreshold must be a number\n")
            usage(parser)
    if args.kmeansclusters:
        try:
            number_of_k_means_clusters = int(args.kmeansclusters)
            print ("Set no of k means clusters: " +
                    str(number_of_k_means_clusters))
        except ValueError:
            print ("kmeansclusters must be a number\n")
            usage(parser)
    if args.shrinkframesize:
        try:
            input_image_reduced_size = int(args.shrinkframesize)
            print ("Set shrinked frame size: " +
                    str(input_image_reduced_size))
        except ValueError:
            print ("shrinkframesize must be a number\n")
            usage(parser)
    if args.maxrequestspersecond:
        try:
            hue_max_requests_per_second = int(args.maxrequestspersecond)
            print ("Set max requests per second: " +
                    str(hue_max_requests_per_second))
        except ValueError:
            print ("maxrequestspersecond must be a number\n")
            usage(parser)
    
    # Auto adjust performance args
    if args.autoadjustperformance:
        try:
            aa = str(args.autoadjustperformance).upper()
            if 'TRUE'.startswith(aa):
                auto_adjust_performance = True
            elif 'FALSE'.startswith(aa):
                auto_adjust_performance = False
            else:
                raise ValueError("autoadjustperformance must be a boolean\n")
            print ("Set auto adjust performance: " + str(auto_adjust_performance))
        except ValueError:
            print ("autoadjustperformance must be a boolean\n")
            usage(parser)
    if args.autoadjustperformancelowfpscount:
        try:
            adjust_counter_limit = int(args.autoadjustperformancelowfpscount)
            print ("Set auto adjust performance low fps count: " +
                    str(adjust_counter_limit))
        except ValueError:
            print ("autoadjustperformancelowfpscount must be a number\n")
            usage(parser)

    # FLICKER args
    if args.flickercorrection:
        try:
            aa = str(args.flickercorrection).upper()
            if 'TRUE'.startswith(aa):
                flicker_prevent = True
            elif 'FALSE'.startswith(aa):
                flicker_prevent = False
            else:
                raise ValueError("flickercorrection must be a boolean\n")
            print ("Set flicker prevent: " + str(flicker_prevent))
        except ValueError:
            print ("flickercorrection must be a boolean\n")
            usage(parser)

    if args.flickerframebuffersize:
        try:
            result_buffer_size = int(args.flickerframebuffersize)
            print ("Set flicker prevent result buffer size : " +
                    str(result_buffer_size))
        except ValueError:
            print ("flickerframebuffersize must be a number\n")
            usage(parser)
    
    if args.flickerresetcounter:
        try:
            flicker_correction_frame_count_until_reset = int(args.flickerresetcounter)
            print ("Set flicker correction frame count until reset : " +
                    str(flicker_correction_frame_count_until_reset))
        except ValueError:
            print ("flickerresetcounter must be a number\n")
            usage(parser)
    
    if args.flickerresetsensitivity:
        try:
            flicker_correction_reset_sensitivity = float(args.flickerresetsensitivity)
            print ("Set flicker correction reset sensitivity : " +
                    str(flicker_correction_reset_sensitivity))
        except ValueError:
            print ("flickerresetsensitivity must be a number\n")
            usage(parser)
    
    if args.colorflickersensitivity:
        try:
            color_flicker_frame_match_threshold = float(args.colorflickersensitivity)
            print ("Set color flicker frame match threshold : " +
                    str(color_flicker_frame_match_threshold))
        except ValueError:
            print ("colorflickersensitivity must be a number\n")
            usage(parser)

    if args.colorflickerinput:
        try:
            color_flicker_prevent_required_input_frames_count = int(args.colorflickerinput)
            print ("Set color flicker detection required input frames count : " +
                    str(color_flicker_prevent_required_input_frames_count))
        except ValueError:
            print ("colorflickerinput must be a number\n")
            usage(parser)

    if args.onoffflickerinput:
        try:
            on_off_flicker_prevent_required_input_frames_count = int(args.onoffflickerinput)
            print ("Set On/Off flicker detection required input frames count : " +
                    str(on_off_flicker_prevent_required_input_frames_count))
        except ValueError:
            print ("onoffflickerinput must be a number\n")
            usage(parser)

    if args.colorflickercount:
        try:
            color_flicker_count_threshold = int(args.colorflickercount)
            print ("Set number of frames that match color flicker condition required to trigger correction : " +
                    str(color_flicker_count_threshold))
        except ValueError:
            print ("colorflickercount must be a number\n")
            usage(parser)
    
    if args.onoffflickercount:
        try:
            on_off_flicker_count_threshold = int(args.onoffflickercount)
            print ("Set number of frames that match on/off flicker condition required to trigger correction : " +
                    str(on_off_flicker_count_threshold))
        except ValueError:
            print ("onoffflickercount must be a number\n")
            usage(parser)
    
    if args.onoffflickerminnzcorrection:
        try:
            min_non_zero_count_on_off_flicker_correction_value = float(args.onoffflickerminnzcorrection)
            print ("Set On/Off flicker min nz count correction : " +
                    str(min_non_zero_count_on_off_flicker_correction_value))
        except ValueError:
            print ("onoffflickerminnzcorrection must be a number\n")
            usage(parser)

    if args.onoffflickerchannelsmincorrection:
        try:
            channels_min_threshold_on_off_flicker_correction_value = float(args.onoffflickerchannelsmincorrection)
            print ("Set On/Off flicker channels min threshold correction : " +
                    str(channels_min_threshold_on_off_flicker_correction_value))
        except ValueError:
            print ("onoffflickerchannelsmincorrection must be a number\n")
            usage(parser)


    # args validation
    if dim_brightness >= starting_brightness:
        print ('dimbrightness must be smaller than maxbrightness')
        usage(parser)
    if transition_type != 0 and transition_type != 1 and transition_type != 2:
        print ('transitiontype must be 0,1 or 2')
        usage(parser)
    if channels_min_threshold >= channels_max_threshold:
        print ('channelsminthreshold must be smaller than channelsmaxthreshold')
        usage(parser)
    if min_non_zero_count >= max_non_zero_count:
        print ('minnzcount must be smaller than maxnzcount')
        usage(parser)
    if min_non_zero_count > 100 \
            or min_non_zero_count < 0 \
            or max_non_zero_count < 0 \
            or max_non_zero_count > 100:
        print ('nz count value must be in interval [0, 100]')
        usage(parser)
    if dim_brightness < 0 or dim_brightness > 256 \
            or starting_brightness < 0 or starting_brightness > 256 \
            or color_skip_sensitivity < 0 or color_skip_sensitivity > 256 \
            or brightness_skip_sensitivity < 0 or brightness_skip_sensitivity > 256 \
            or channels_min_threshold < 0 or channels_min_threshold > 256 \
            or channels_max_threshold < 0 or channels_max_threshold > 256:
        print ('dimbrightness, maxbrightness, colorskipsensitivity\
               channelsminthreshold, channelsmaxthreshold values must be in interval [0, 256]')
        usage(parser)

    # LIGHTS VALIDATION
    number_of_lights = len(my_light_ids) | len(my_light_names)
    if (number_of_lights == 0):
        print ('Please select at least one light.')
        usage(parser)

    if not dim_lights_instead_of_turn_off:
        dim_brightness = 3

    # Variables
    go_dark = False
    current_brightness = starting_brightness
    current_transition_time = 0
    if transition_type == 2:
        current_transition_time = transition_time
    prev_color = None
    prev_frame = None
    prev_brightness = None
    prev_fps = None

    # Auto adjust performance vars
    adjust_position = 0
    adjust_counter = 0
    request_timeout = 1/hue_max_requests_per_second * number_of_lights

    # FLICKER prevent vars
    result_buffer = []
    color_flicker_countdown = 0
    on_off_flicker_countdown = 0

    # FLICKER correction defaults
    min_non_zero_count_flicker_correction = 1
    channels_min_threshold_flicker_correction = 1
    k_means_flicker_correction = 0

    global CAN_UPDATE_HUE

    # If the app is not registered and the button is not pressed,
    # press the button and call connect() (this only needs to be run a single time)

    # Your bridge IP
    connected = False
    shown_instructions = False
    current_dir = os.path.dirname(__file__)
    phue_config_file = os.path.join(current_dir, PHUE_CONFIG_FILE)
    bridge_ip = DISCOVERY_LIB.getBridgeIP()
    print ("Discovered bridge ip: " + bridge_ip)
    print('Connecting to bridge')
    i = 0
    while i < 30:
        time.sleep(1)
        try:
            bridge = Bridge(bridge_ip, None, phue_config_file)
        except PhueRegistrationException:
            if not shown_instructions:
                print('Press the Hue Bridge button in order to register')
                shown_instructions = True
            i += 1
            continue
        else:
            connected = True
            break
    if not connected:
        print('Failed to register to bridge')
        sys.exit()

    bridge.connect()

    print ('Connected to Hue Bridge with address {0}'.format(bridge_ip))

    if (user == None):
        user = bridge.username

    light_names = bridge.get_light_objects('name')

    # Init lights
    if not my_light_ids:
        for hue_light in my_light_names:
            light_names[hue_light].on = True
            light_names[hue_light].brightness = starting_brightness
            my_light_ids.append(light_names[hue_light].light_id)
    else:
        for hue_light_id in my_light_ids:
            bridge.set_light(hue_light_id, 'on', True)
            bridge.set_light(hue_light_id, 'bri', starting_brightness)
           
    with mss.mss() as sct:
        # Part of the screen to capture (use if you want to create a multiple color effect)
        full_mon=sct.monitors[1]
        monitor = full_mon
        
        if screen_part_to_capture == "full":
            monitor = full_mon
        elif screen_part_to_capture == "left":
            half_mon_width = int(full_mon["width"]/2)
            monitor = {'top': 0, 'left': 0, 'width': half_mon_width, 'height': full_mon["height"]}
        elif screen_part_to_capture == "right":
            half_mon_width = int(full_mon["width"]/2)
            monitor = {'top': 0, 'left': half_mon_width, 'width': half_mon_width, 'height': full_mon["height"]}
        elif screen_part_to_capture == "side-left":
            side_mon_width = int(full_mon["width"]/4)
            monitor = {'top': 0, 'left': 0, 'width': side_mon_width, 'height': full_mon["height"]}
        elif screen_part_to_capture == "side-right":
            side_mon_width = int(full_mon["width"]/4)
            side_mon_right_offset = full_mon["width"] - side_mon_width
            monitor = {'top': 0, 'left': side_mon_right_offset, 'width': side_mon_width, 'height': full_mon["height"]}



        while 'Screen capturing':
                        
            last_time = time.time()
            
            # Get raw pixels from the screen, save it to a Numpy array
            img = numpy.array(sct.grab(monitor))

            # Shrink image for performance sake
            current_frame = FRAME_COLOR_LIB.shrink_image(img, input_image_reduced_size)
            
            # init frame comparrison result
            comparison_result = None

            # Compare Frame with Prev Frame
            # Skip if similar
            if prev_frame is not None:
                
                comparison_result = cv2.matchTemplate(
                    current_frame, prev_frame, 1)
                if comparison_result[0][0] < frame_match_sensitivity:
                    continue

                #transition stuff
                elif transition_type == 1:
                    if comparison_result[0][0] < frame_transition_sensitivity:
                        current_transition_time = transition_time
                    else:
                        current_transition_time = 0
            
            # Apply dark color threshold and compute mask
            masked_frame = FRAME_COLOR_LIB.apply_frame_mask(current_frame, channels_min_threshold * channels_min_threshold_flicker_correction)

            current_brightness = FRAME_COLOR_LIB.calculate_frame_brightness(masked_frame, 
                dim_brightness, starting_brightness, min_non_zero_count * min_non_zero_count_flicker_correction,
                max_non_zero_count)
            
            # Turn on/off depending on result brightness
            if not dim_lights_instead_of_turn_off:
                if current_brightness <= dim_brightness:
                    go_dark = True
                else:
                    go_dark = False

            # Calculate relevant color for this frame
            result_color = FRAME_COLOR_LIB.calculate_hue_color(
                masked_frame, (number_of_k_means_clusters-k_means_flicker_correction), color_spread_threshold,
                channels_min_threshold * channels_min_threshold_flicker_correction, channels_max_threshold)
            
            # save brightness, go_dark and diff from prev frame for later use
            result_color.brightness = current_brightness
            result_color.go_dark = go_dark
            if(comparison_result is not None):
                result_color.diff_from_prev = comparison_result[0][0]

            # Compare Current Calculated Color with previous Color
            # Skip frame if result color is almost identical
            if prev_color is not None and\
                abs(current_brightness - prev_brightness) < brightness_skip_sensitivity:
                skip_frame = True
                for j in range(0, 3):
                    ch_diff = math.fabs(
                        int(prev_color.color[j]) - int(result_color.color[j]))
                    if ch_diff > color_skip_sensitivity:
                        skip_frame = False
                        break
                if skip_frame:
                    continue
            
            # Anti Flicker algorithm 
            # BETA VERSION
            if(flicker_prevent):
                result_buffer.append(result_color)
                current_buffer_size = len(result_buffer)

                # reset flicker temporary adjustments if any
                if(result_color.diff_from_prev is not None and \
                    result_color.diff_from_prev > flicker_correction_reset_sensitivity):
                    if(k_means_flicker_correction != 0):
                        color_flicker_countdown = 0
                        k_means_flicker_correction = 0
                        print('reset color FLICKER adjustments')
                    if(min_non_zero_count_flicker_correction != 1):
                        on_off_flicker_countdown = 0
                        min_non_zero_count_flicker_correction = 1
                        channels_min_threshold_flicker_correction = 1
                        print('reset on/off FLICKER adjustments')
                else:                    
                    color_flicker_count = 0
                    current_buffer_size = len(result_buffer)
                    if current_buffer_size >= color_flicker_prevent_required_input_frames_count:
                        for j in range (current_buffer_size - (color_flicker_prevent_required_input_frames_count), \
                                        current_buffer_size-1):
                            color_check =  FRAME_COLOR_LIB.frame_colors_are_similar(\
                                            result_buffer[j], result_buffer[j-1],
                                            color_skip_sensitivity, brightness_skip_sensitivity)
                            if result_buffer[j].diff_from_prev is not None and\
                                result_buffer[j].diff_from_prev < color_flicker_frame_match_threshold and\
                                not color_check:
                                color_flicker_count += 1

                        if color_flicker_count >= color_flicker_count_threshold:
                            # Lower the k means if color flicker detected
                            k_means_flicker_correction = int(number_of_k_means_clusters/2) - 1
                            print('COLOR FLICKER AVOIDED | new K: {0}'.format(number_of_k_means_clusters - k_means_flicker_correction))
                            color_flicker_countdown = flicker_correction_frame_count_until_reset                    

                    # ON OFF FLICKER
                    if not dim_lights_instead_of_turn_off and go_dark:
                        on_off_count = 0
                        current_buffer_size = len(result_buffer)
                        if current_buffer_size >= on_off_flicker_prevent_required_input_frames_count:
                            for j in range (current_buffer_size - (on_off_flicker_prevent_required_input_frames_count + 1), \
                                            current_buffer_size-1):
                                if not result_buffer[j].go_dark and result_buffer[j+1].go_dark:
                                    on_off_count += 1

                            if on_off_count >= on_off_flicker_count_threshold:
                                # Lower the turn off threshold if lights are switched off many times in a short period of time
                                min_non_zero_count_flicker_correction = min_non_zero_count_on_off_flicker_correction_value
                                channels_min_threshold_flicker_correction = channels_min_threshold_on_off_flicker_correction_value
                                print('ON/OFF FLICKER AVOIDED | new min nz: {0}'.format(min_non_zero_count * min_non_zero_count_flicker_correction))
                                on_off_flicker_countdown = flicker_correction_frame_count_until_reset

                    # Keep buffer at specified size
                    if len(result_buffer) == result_buffer_size:
                        result_buffer.pop(0)

                    if color_flicker_countdown >= 1:
                        color_flicker_countdown -= 1
                        if color_flicker_countdown == 0:
                            k_means_flicker_correction = 0
                            print('back from color FLICKER')

                    if on_off_flicker_countdown >= 1:
                        on_off_flicker_countdown -= 1
                        if on_off_flicker_countdown == 0:
                            min_non_zero_count_flicker_correction = 1
                            channels_min_threshold_flicker_correction = 1
                            print('back from on off FLICKER')

                
                    # print('RESULT_BUFFER count: {0}'.format(len(RESULT_BUFFER)))

            # Send color to Hue if update flag is clear
            if CAN_UPDATE_HUE:

                # Use prev color if RGB are all 0
                # Avoids unpleasant blue shift
                if result_color.color[0] == 0 and result_color.color[1] == 0\
                    and result_color.color[2] == 0:
                    if prev_color:
                        result_color = prev_color
                    else:
                        result_color.color[0] = DEFAULT_DARK_COLOR[0]
                        result_color.color[1] = DEFAULT_DARK_COLOR[1]
                        result_color.color[2] = DEFAULT_DARK_COLOR[2]
                else:
                    prev_color = result_color

                prev_frame = current_frame
                prev_brightness = current_brightness

                # Timer that limits the requests to hue bridge in order to prevent bottlenecks
                CAN_UPDATE_HUE = False
                update_timer = threading.Timer(
                    request_timeout, clear_update_flag)
                update_timer.start()

                print ('Updating with RGB: [{0}, {1}, {2}]'.format(
                    result_color.color[0], result_color.color[1], result_color.color[2]))
                print ('brightness: {0}'.format(current_brightness))

                if dim_lights_instead_of_turn_off:
                    command = {'xy': result_color.get_hue_color(),
                                'bri': current_brightness,
                                'transitiontime': current_transition_time}
                else: 
                    command = {
                                'on': not go_dark,
                                'xy': result_color.get_hue_color(),
                                'bri': current_brightness,
                                'transitiontime': current_transition_time}
                
                # Slower request possibilities at the time of writing:
                #bridge.set_light(my_light_names, command)
                #bridge.set_group(group_id, command)
                # r = requests.put('http://%s/api/%s/groups/%s/action'%(bridge_ip, user, group_id), data = json.dumps(command))
                
                for hue_light_id in my_light_ids:
                    requests.put('http://%s/api/%s/lights/%s/state'%(bridge_ip, user, hue_light_id), data = json.dumps(command))
                    #print(r.text)

                prev_fps = 1 / (time.time()-last_time)
                print ('fps: {0}'.format(prev_fps))

                # BETA VERSION !!!
                # If bad FPS
                # Adjust params to achieve smooth performance 
                if(auto_adjust_performance 
                    and prev_fps 
                    and prev_fps < hue_max_requests_per_second):
                    # Skip random bad fps frames
                    if(adjust_counter == adjust_counter_limit - 1):
                        if adjust_position % 6 < 3  and input_image_reduced_size >= 30:
                            print("adjust shrink image size")
                            input_image_reduced_size -= 10
                        elif adjust_position % 6 < 3:
                            # increment if there is nothing to do here
                            adjust_position = 3

                        if adjust_position % 6 == 3 and number_of_k_means_clusters > 4:
                            print("adjust k means")
                            number_of_k_means_clusters -= 1
                        elif adjust_position % 6 == 3:
                            # increment if there is nothing to do here
                            adjust_position += 1

                        if adjust_position % 6 == 4 and channels_min_threshold < 90:
                            print("adjust channels_min_threshold")
                            channels_min_threshold += 5
                        elif adjust_position % 6 == 4:
                            # increment if there is nothing to do here
                            adjust_position += 1

                        if adjust_position % 6 == 5 and frame_match_sensitivity < 0.1:
                            print("adjust match sensitivity")
                            frame_match_sensitivity += 0.001
                        elif adjust_position % 6 == 5:
                            # increment if there is nothing to do here
                            adjust_position += 1

                        adjust_position += 1
                        adjust_counter = 0
                        print("adjusted params to meet performance")
                        print ('input_image_reduced_size: {0}'.format(input_image_reduced_size))
                        print ('channels_min_threshold: {0}'.format(channels_min_threshold))
                        print ('number_of_k_means_clusters: {0}'.format(number_of_k_means_clusters))
                        print ('frame_match_sensitivity: {0}'.format(frame_match_sensitivity))
                    else:
                        adjust_counter += 1
                else:
                    adjust_counter = 0

if __name__ == "__main__":
    main(sys.argv[1:])
