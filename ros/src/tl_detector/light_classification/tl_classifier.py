from styx_msgs.msg import TrafficLight
import cv2
from keras.models import load_model
from numpy import newaxis
import numpy as np
import tensorflow as tf 
import os

class TLClassifier(object):
    def __init__(self):
	path = os.getcwd()
        self.model = load_model(path + '/light_classification/model.h5') 
	self.model._make_predict_function()
        self.graph = tf.get_default_graph()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        image = cv2.resize(image, (400, 400)) 
        image = image.astype(float)
    	image = image / 255.0
    	image = image[newaxis,:,:,:]
	with self.graph.as_default():
            predictions = self.model.predict(image)
    	classification = np.argmax(predictions, axis=1)


        if(classification[0] == 1):
           return TrafficLight.RED


        return TrafficLight.UNKNOWN