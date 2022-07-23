from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.resnet import ResNet50
from keras.preprocessing.image import ImageDataGenerator

from keras_retinanet import models
from glob import glob
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf
# re-size all the images to this


valid_path = './classes/cep/p1/photos/1/objectDetection/' #cropped
#valid_path = '../predict/test'  #not cropped

test_datagen = ImageDataGenerator(rescale=1./255)


test_images=test_datagen.flow_from_directory(valid_path,
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='categorical')
model = models.load_model('./resnet50_pascal_30.h5', backbone_name='resnet50')
model = models.convert_model(model)
#vgg19 = load_model('./Finalmodels/retrainvgg19.h5')
#resnet= load_model('./Finalmodels/retrainResnet.h5')
#cnn= load_model('./Finalmodels/retrainCNN.h5')



lossvgg16, accvgg16 = model.evaluate(test_images, verbose=2)

#lossvgg19, accvgg19 = vgg19.evaluate(test_images, verbose=2)

#lossresnet, accresnet = resnet.evaluate(test_images, verbose=2)

#losscnn, acccnn = cnn.evaluate(test_images, verbose=2)


print("Restored model, vgg16 accuracy: {:5.2f}%".format(100 * accvgg16))
#print("Restored model, vgg19 accuracy: {:5.2f}%".format(100 * accvgg19))
#print("Restored model, resnet accuracy: {:5.2f}%".format(100 * accresnet))
#print("Restored model, cnn accuracy: {:5.2f}%".format(100 * acccnn))

