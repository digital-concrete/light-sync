# hue-ambiance
Sync lights with screen in real time

How it works:

1/ Use Python MSS to grab a screen capture

2/ Shrink the captured image in order to increase computation speed

3/ Apply OpenCV K Means in order to find dominant colors

4/ Find out which dominant color is most present on the screen

5/ Set hue lights to use that color


Based on:

http://python-mss.readthedocs.io/examples.html

https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html

https://github.com/studioimaginaire/phue

https://github.com/benknight/hue-python-rgb-converter

