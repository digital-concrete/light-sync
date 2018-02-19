import time
import random
import cv2
import mss
import numpy

from phue import Bridge
from convertorlib import Converter

converter = Converter()

# Your bridge IP
b = Bridge('192.168.1.22')

# If the app is not registered and the button is not pressed, press the button and call connect() (this only needs to be run a single time)
b.connect()

# Get the bridge state (This returns the full dictionary that you can explore)
b.get_api()


with mss.mss() as sct:
    # Part of the screen to capture (use if you want to create a multiple color effect)
    #monitor = {'top': 40, 'left': 0, 'width': 800, 'height': 640}
    monitor = sct.monitors[1]

    while 'Screen capturing':
        last_time = time.time()

        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.array(sct.grab(monitor))


        #Reduce image size to increase computation speed
        height, width = img.shape[:2]
        max_height = 600
        max_width = 600

        if max_height < height or max_width < width:
            # get scaling factor
            scaling_factor = max_height / float(height)
            if max_width/float(width) < scaling_factor:
                scaling_factor = max_width / float(width)
            # resize image
            img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        #COMPUTE K MEANS
        Z = img.reshape((-1,4))

        # convert to np.float32
        Z = numpy.float32(Z)

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 8
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = numpy.uint8(center)

        #Compute most prolific mean
        
        labelcounts = []
        for i in range(K):
            labelcounts.append(0)
        for x in range(0, label.size):
            labelcounts[label[x][0]]+=1

        print(labelcounts)
        maxlabel = labelcounts[0]
        maxlabelindex = 0    
        for j in range(1, len(labelcounts)):
            if labelcounts[j] > maxlabel:
                maxlabelindex = j
                maxlabel = labelcounts[j] 

        #Set Hue lights to use this color
        dominantcolor = center[maxlabelindex]
        light_names = b.get_light_objects('name')

        # Replace with your light names
        for light in ['Light1', 'Light2']:
            #Set brightness
            #light_names[light].brightness = 50

            #Set color with the help of convertor
            light_names[light].xy = converter.rgb_to_xy(dominantcolor[2], dominantcolor[1], dominantcolor[0])
        
        print('fps: {0}'.format(1 / (time.time()-last_time)))
