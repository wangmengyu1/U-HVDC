# -*- coding: utf-8 -*-
"""
Apply the test set to the trained model for prediction and calculate various evaluation metrics.
"""

from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
import os
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from sklearn import preprocessing as prep

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Test data directories
test_geo_path = './data/newsamplemark811/1418_test_geo/'  # Test set data
test_label_path = './data/newsamplemark811/1418_test_label/'  # Test set labels
e_code = "FE8"  # Experiment code

test_geo_files = os.listdir(test_geo_path)  # List of test data files
test_label_files = os.listdir(test_label_path)  # List of test label files

label_test = []  # Store the true labels of the test set
label_pred = []  # Store the predicted labels by the model

# Load the model for this test
print("e_code={}".format(e_code))
# model = load_model('./results/model/{}/bmodel.h5'.format(e_code))  # Load the best performing model on validation set (max val_acc)
model = load_model('./image1/{}/bmodel.h5'.format(e_code))  # Load the best performing model on validation set (max val_acc)

if len(test_geo_files) == len(test_label_files):
    for i in range(len(test_geo_files)):
        select = test_geo_files[i]
        if test_geo_files[i] == test_label_files[i]:
            a_geo = np.load(test_geo_path + select)  # Load the test data
            a_label = np.load(test_label_path + select)  # Load the test labels
            a_label_list = a_label.tolist()  # Convert to list
            label_test.extend(a_label_list)  # Add to true labels
            K.clear_session()  # Clear Keras session
            tf.compat.v1.reset_default_graph()  # Reset TensorFlow graph
            a_geo = np.expand_dims(prep.scale(a_geo), axis=1)  # Scale and expand dimensions of input
            a_geo = np.expand_dims(a_geo, axis=0)
            a_pred = model.predict(a_geo)  # Get predictions from the model
            a_pred_array = []
            list_pred = a_pred[0].tolist()  # Convert predictions to list
            for j in range(len(list_pred)):
                prob_background = list_pred[j][0]  # Probability of background
                prob_hvdc = list_pred[j][1]  # Probability of hvdc
                if prob_hvdc >= 0.5:
                    a_pred_array.append(1.0)  # Append 1 if prob_hvdc >= 0.5
                else:
                    a_pred_array.append(0.0)  # Append 0 otherwise
            label_pred.extend(a_pred_array)  # Add predictions to the list
            if i % 100 == 0:
                print("{}/{}".format(i + 1, len(test_geo_files)))  # Print progress
        else:
            print("Names do not match")
            print("break")
            break

print("len true", len(label_test))  # Print length of true labels
print("len pred", len(label_pred))  # Print length of predicted labels
acc_nums = accuracy_score(label_test, label_pred, normalize=False)  # Calculate total correct predictions
print("acc_nums", acc_nums)
# Accuracy
print("{}".format(e_code))
acc = accuracy_score(label_test, label_pred)  # Calculate accuracy
print("acc", acc)
# Precision
p = precision_score(label_test, label_pred)  # Calculate precision
print("precision", p)
# Recall
r = recall_score(label_test, label_pred)  # Calculate recall
print("recall", r)
# F1 Score
f1 = f1_score(label_test, label_pred)  # Calculate F1 score
print("f1", f1)
