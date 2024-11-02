Data Files

1418_train_geo1.zip, 1418_train_geo2.zip, 1418_train_geo3.zip:

These three zip files contain the training data for the model. Due to the large size of the training dataset, it has been split into three separate files for easier upload and handling.

1418_val_geo.zip, 1418_val_label.zip:

These zip files contain the validation dataset:

1418_val_geo.zip includes the validation data.

1418_val_label.zip includes the labels for the validation data.

1418_test_geo.zip, 1418_test_label.zip

These zip files contain the test dataset:

1418_test_geo.zip includes the test data.

1418_test_label.zip includes the labels for the test data.

Code Files

LoadBatches1D.py:

This file contains the data generator for loading data in batches. It is used to efficiently load large datasets during training, helping to manage memory usage.

U-TSS.py:

This file contains the main structure of the U-TSS model, which is based on the U-Net architecture adapted for time series segmentation.

train.py:

This file is responsible for training the U-TSS model and includes functionality to save the best model weights based on performance during training.

Score_calculation.py:

This file is used to evaluate the model's performance on the test set. It calculates various evaluation metrics to assess the model's effectiveness on the test data.
