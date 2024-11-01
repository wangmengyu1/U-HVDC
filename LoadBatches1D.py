"""
LoadBatches1D.py: Iteratively generates batches for training

"""

import os
import itertools
import numpy as np
from sklearn import preprocessing as prep

# Previously forgot to modify the corresponding parameters here

def getSigArr(path, sigNorm='scale'):
    sig = np.load(path)  # Load the signal data from the specified path
    if sigNorm == 'scale':
        sig = prep.scale(sig)  # Scale the signal using standardization
    elif sigNorm == 'minmax':
        min_max_scaler = prep.MinMaxScaler()  # Initialize Min-Max scaler
        sig = min_max_scaler.fit_transform(sig)  # Scale the signal to the range [0, 1]
    return np.expand_dims(sig, axis=1)  # Add an extra dimension to the signal array

# Modification area
def getSegmentationArr(path, nClasses=2, output_length=1440, class_value=[0, 1]):  
    # class_value is defined in generate_labels.py; background is 0, normal is 0.5, premature beat is 1; must remain consistent
    # class_value is defined in generate_labels.py; background is 0, HVDC is 1; must remain consistent
    seg_labels = np.zeros([output_length, nClasses])  # Initialize an array for segmentation labels
    seg = np.load(path)  # Load the segmentation data from the specified path
    for i in range(nClasses):
        seg_labels[:, i] = (seg == class_value[i]).astype(float)  # Assign values to segmentation labels based on class values
    return seg_labels  # Return the segmentation labels

def SigSegmentationGenerator(sigs_path, segs_path, batch_size, n_classes, output_length=1440):  # 1440 -- 86400
    sigs = os.listdir(sigs_path)  # List all signal files in the specified path
    segmentations = os.listdir(segs_path)  # List all segmentation files in the specified path
    sigs.sort()  # Sort the signal file names
    segmentations.sort()  # Sort the segmentation file names
    for i in range(len(sigs)):
        sigs[i] = sigs_path + sigs[i]  # Create the full path for each signal file
        segmentations[i] = segs_path + segmentations[i]  # Create the full path for each segmentation file
    assert len(sigs) == len(segmentations)  # Ensure the number of signals matches the number of segmentations
    for sig, seg in zip(sigs, segmentations):
        assert (sig.split('/')[-1].split(".")[0] == seg.split('/')[-1].split(".")[0])  # Ensure file names match
    zipped = itertools.cycle(zip(sigs, segmentations))  # Create an infinite cycle iterator for signal-segmentation pairs
    while True:
        X = []  # Initialize the input batch
        Y = []  # Initialize the output batch
        for _ in range(batch_size):
            sig, seg = next(zipped)  # Get the next pair of signal and segmentation files
            X.append(getSigArr(sig))  # Get the processed signal array and append it to the input batch
            Y.append(getSegmentationArr(seg, n_classes, output_length))  # Get the segmentation labels and append to the output batch
        yield np.array(X), np.array(Y)  # Yield the input and output batches as numpy arrays
