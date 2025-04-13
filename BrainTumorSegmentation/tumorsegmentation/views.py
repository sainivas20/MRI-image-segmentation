from django.shortcuts import render
from tumorsegmentation.constants import modelpath,uploadpath

from keras.models import load_model
from keras.losses import binary_crossentropy
from keras import backend as K

import tensorflow as tf

import cv2

import numpy as np
import os

import matplotlib.pyplot as plt

from tumorsegmentation.forms import ImageForm
from tumorsegmentation.models import ImageModel

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def get_iou_vector(A, B):
    t = A>0
    p = B>0
    intersection = np.logical_and(t,p)
    union = np.logical_or(t,p)
    iou = (np.sum(intersection) + 1e-10 )/ (np.sum(union) + 1e-10)
    return iou

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def iou_metric(label, pred):
    return tf.py_func(get_iou_vector, [label, pred > 0.5], tf.float64)

def makepredict(image):

    unet = load_model(modelpath+'\\model_best_checkpoint.h5',custom_objects={'bce_dice_loss': bce_dice_loss, 'iou_metric': iou_metric})

    x_test = []

    image_shape = (128, 128)

    img = cv2.imread(os.path.join(uploadpath, image))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # ADD THIS
    img = cv2.resize(gray, image_shape)
    x_test.append(img)

    x_test = np.array(x_test)  # ADD THIS
    x_test = x_test.reshape(x_test.shape[0], 128, 128, 1)  # ADD THIS

    print(type(x_test), x_test.shape)

    THRESHOLD = 0.2
    predicted_mask = (unet.predict(x_test) > THRESHOLD) * 1
    print(type(predicted_mask), predicted_mask.shape)

    return predicted_mask

def predict(request):

    imageForm = ImageForm(request.POST, request.FILES)

    if imageForm.is_valid():

        imageModel = ImageModel()
        imageModel.photo = imageForm.cleaned_data["photo"]
        imageModel.save()

        try:
            last=ImageModel.objects.last()
            image =str(last.photo).split("/")[1]

            predicted_mask = makepredict(image)
            print(type(predicted_mask), predicted_mask.shape)

            plt.imshow(predicted_mask[0])
            plt.title("Predicted Mask");
            plt.savefig(uploadpath+"output.jpg")

            return render(request, 'result.html',{"result":"output.jpg"})

        except Exception as e:
            print("Exception:",e)
            return render(request, 'index.html', {"message": "Prediction Failed"})

    return render(request, 'index.html', {"message": "Invalid Form"})