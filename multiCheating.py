from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import tensorflow as tf
import sys
######################################
sys.path.insert(0, '../')
tf.get_logger().setLevel('ERROR')

######objectDetection
objmodel_path = os.path.join('./resnet50_coco_best_v2.1.0.h5')
objmodel = models.load_model(objmodel_path, backbone_name='resnet50')

def objectDetection(frame):
    

    labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
                    39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}


    draw = frame.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    frame = preprocess_image(frame)
    frame, scale = resize_image(frame)
    boxes, scores, labels = objmodel.predict_on_batch(
        np.expand_dims(frame, axis=0))

    boxes /= scale
    predictions =[]
    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5:
            break

        color = label_color(label)

        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)
        predictions.append(caption)

    return predictions


webcam = cv2.VideoCapture('./WIN_20220228_11_51_45_Pro.mp4')

counter =0
while webcam.isOpened():

    counter=counter+1
    status, frame = webcam.read()
    
    if counter%10 ==0:
        #hp = headpose(frame)
        od = objectDetection(frame)
        #eg = eyeGaze(frame)
        #print('headpose', hp)
        print('objectDetection', od)
       # print('eyeGaze',eg)
        cv2.imshow("cheating detection", frame)
        counter =0
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# release resources
webcam.release()
cv2.destroyAllWindows()