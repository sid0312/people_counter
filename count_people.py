# -*- coding: utf-8 -*-
"""
Created on Sun May 17 23:15:10 2020

@author: siddh
"""
import mxnet as mx
import time 
import gluoncv as gcv
from gluoncv import model_zoo
import cv2 

model_used = 'yolo3_darknet53_coco'

def convert_to_mxnet(frame):
  return mx.nd.array(frame)

def transform_image(array):
    t = gcv.data.transforms.presets.yolo.transform_test(array,short=416)
    return t

def detect(network,data):
  prediction = network(data)
  class_ids, scores, bounding_boxes = prediction
  return class_ids, scores, bounding_boxes

def count_objects(network, class_ids, scores, bounding_boxes, object_label, threshold=0.5):
    num_people = 0
    classes = network.classes
    scores = scores.asnumpy().squeeze().astype(float)
    class_ids = class_ids.asnumpy().squeeze().astype(int)
    for index,i in enumerate(class_ids):
        if classes[i]==object_label:
            if scores[index]>=threshold:
                num_people+=1   
    return num_people

class People_Counter():
    def __init__(self):
        self._network = model_zoo.get_model(model_used, pretrained=True)
        self._network.hybridize()

    def set_threshold(self, threshold):
        self._threshold = threshold
        
    def count(self, frame):
        norm_image,unnorm_image = transform_image(convert_to_mxnet(frame))
        class_ids, scores, bounding_boxes = detect(self._network,norm_image)
        num_people = count_objects(self._network,class_ids,scores,bounding_boxes,object_label='person',threshold=self._threshold)
        return num_people
    def detect_people(self,frame):
        norm_image,unnorm_image = transform_image(convert_to_mxnet(frame))
        return detect(self._network,norm_image)
    def class_names(self):
        return self._network.classes

counter = People_Counter()
counter.set_threshold(0.5)
video = cv2.VideoCapture(0)
time.sleep(1)
while True:
    _,frame = video.read()
    num_people = counter.count(frame)
    if num_people == 1:
        text= '{} person detected.'.format(num_people)
    else:
        text = '{} people detected.'.format(num_people)
    print(text)
    color = (0,0,255)
    thickness=2
    fontScale=1
    org = (120,450)
    font = cv2.FONT_HERSHEY_SIMPLEX
    countme = cv2.putText(frame, text, org, font, fontScale,color, thickness, cv2.LINE_AA, False) 
    
    class_ids,scores,bounding_boxes = counter.detect_people(frame)
    preds = gcv.utils.viz.cv_plot_bbox(frame,bounding_boxes[0],scores[0],class_ids[0],class_names = counter.class_names())
    cv2.imshow('result',countme)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
video.release()
cv2.destroyAllWindows() 


