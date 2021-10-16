"""
Title: Speaker Recognition
Author: [Fadi Badine](https://twitter.com/fadibadine)
Date created: 14/06/2020
Last modified: 03/07/2020
Description: Classify speakers using Fast Fourier Transform (FFT) and a 1D Convnet.
"""
"""
## Introduction

This example demonstrates how to create a model to classify speakers from the
frequency domain representation of speech recordings, obtained via Fast Fourier
Transform (FFT).

It shows the following:

- How to use `tf.data` to load, preprocess and feed audio streams into a model
- How to create a 1D convolutional network with residual
connections for audio classification.

Our process:

- We prepare a dataset of speech samples from different speakers, with the speaker as label.
- We add background noise to these samples to augment our data.
- We take the FFT of these samples.
- We train a 1D convnet to predict the correct speaker given a noisy FFT speech sample.

Note:

- This example should be run with TensorFlow 2.3 or higher, or `tf-nightly`.
- The noise samples in the dataset need to be resampled to a sampling rate of 16000 Hz
before using the code in this example. In order to do this, you will need to have
installed `ffmpg`.
"""

"""
## Setup
"""

import os
import shutil
import numpy as np

import tensorflow as tf
from tensorflow import keras

from pathlib import Path
from IPython.display import display, Audio

# Get the data from https://www.kaggle.com/kongaevans/speaker-recognition-dataset/download
# and save it to the 'Downloads' folder in your HOME directory
DATASET_ROOT = os.path.join(os.path.expanduser("~"), "Downloads/16000_pcm_speeches")

# The folders in which we will put the audio samples and the noise samples
AUDIO_SUBFOLDER = "audio"
NOISE_SUBFOLDER = "noise"

DATASET_AUDIO_PATH = os.path.join(DATASET_ROOT, AUDIO_SUBFOLDER)
DATASET_NOISE_PATH = os.path.join(DATASET_ROOT, NOISE_SUBFOLDER)

# Percentage of samples to use for validation
VALID_SPLIT = 0.1

# Seed to use when shuffling the dataset and the noise
SHUFFLE_SEED = 43

# The sampling rate to use.
# This is the one used in all of the audio samples.
# We will resample all of the noise to this sampling rate.
# This will also be the output size of the audio wave samples
# (since all samples are of 1 second long)
SAMPLING_RATE = 16000

# The factor to multiply the noise with according to:
#   noisy_sample = sample + noise * prop * scale
#      where prop = sample_amplitude / noise_amplitude
SCALE = 0.5

BATCH_SIZE = 128
EPOCHS = 100

'''
This script performs the basic process for applying a machine learning
algorithm to a dataset using Python libraries.

The four steps are:
   1. Download a dataset (using pandas)
   2. Process the numeric data (using numpy)
   3. Train and evaluate learners (using scikit-learn)
   4. Plot and compare results (using matplotlib)


The data is downloaded from URL, which is defined below. As is normal
for machine learning problems, the nature of the source data affects
the entire solution. When you change URL to refer to your own data, you
will need to review the data processing steps to ensure they remain
correct.

============
Example Data
============
The example is from http://mlr.cs.umass.edu/ml/datasets/Spambase
It contains pre-processed metrics, such as the frequency of certain
words and letters, from a collection of emails. A classification for
each one indicating 'spam' or 'not spam' is in the final column.
See the linked page for full details of the data set.

This script uses three classifiers to predict the class of an email
based on the metrics. These are not representative of modern spam
detection systems.
'''

# Remember to update the script for the new data when you change this URL
URL = "http://mlr.cs.umass.edu/ml/machine-learning-databases/spambase/spambase.data"

# Uncomment this call when using matplotlib to generate images
# rather than displaying interactive UI.
#import matplotlib
#matplotlib.use('Agg')

from pandas import read_table
import numpy as np
import matplotlib.pyplot as plt

try:
    # [OPTIONAL] Seaborn makes plots nicer
    import seaborn
except ImportError:
    pass

# =====================================================================

def download_data():
    '''
    Downloads the data for this script into a pandas DataFrame.
    '''

    # If your data is in an Excel file, install 'xlrd' and use
    # pandas.read_excel instead of read_table
    #from pandas import read_excel
    #frame = read_excel(URL)

    # If your data is in a private Azure blob, install 'azure-storage' and use
    # BlockBlobService.get_blob_to_path() with read_table() or read_excel()
    #from azure.storage.blob import BlockBlobService
    #service = BlockBlobService(ACCOUNT_NAME, ACCOUNT_KEY)
    #service.get_blob_to_path(container_name, blob_name, 'my_data.csv')
    #frame = read_table('my_data.csv', ...

    frame = read_table(
        URL,
        
        # Uncomment if the file needs to be decompressed
        #compression='gzip',
        #compression='bz2',

        # Specify the file encoding
        # Latin-1 is common for data from US sources
        encoding='latin-1',
        #encoding='utf-8',  # UTF-8 is also common

        # Specify the separator in the data
        sep=',',            # comma separated values
        #sep='\t',          # tab separated values
        #sep=' ',           # space separated values

        # Ignore spaces after the separator
        skipinitialspace=True,

        # Generate row labels from each row number
        index_col=None,
        #index_col=0,       # use the first column as row labels
        #index_col=-1,      # use the last column as row labels

        # Generate column headers row from each column number
        header=None,
        #header=0,          # use the first line as headers

        # Use manual headers and skip the first row in the file
        #header=0,
        #names=['col1', 'col2', ...],
    )

    # Return a subset of the columns
    #return frame[['col1', 'col4', ...]]

    # Return the entire frame
    return frame


# =====================================================================


def get_features_and_labels(frame):
    '''
    Transforms and scales the input data and returns numpy arrays for
    training and testing inputs and targets.
    '''

    # Replace missing values with 0.0, or we can use
    # scikit-learn to calculate missing values (below)
    #frame[frame.isnull()] = 0.0

    # Convert values to floats
    arr = np.array(frame, dtype=np.float)

    # Use the last column as the target value
    X, y = arr[:, :-1], arr[:, -1]
    # To use the first column instead, change the index value
    #X, y = arr[:, 1:], arr[:, 0]
    
    # Use 80% of the data for training; test against the rest
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # sklearn.pipeline.make_pipeline could also be used to chain 
    # processing and classification into a black box, but here we do
    # them separately.
    
    # If values are missing we could impute them from the training data
    #from sklearn.preprocessing import Imputer
    #imputer = Imputer(strategy='mean')
    #imputer.fit(X_train)
    #X_train = imputer.transform(X_train)
    #X_test = imputer.transform(X_test)
    
    # Normalize the attribute values to mean=0 and variance=1
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # To scale to a specified range, use MinMaxScaler
    #from sklearn.preprocessing import MinMaxScaler
    #scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit the scaler based on the training data, then apply the same
    # scaling to both training and test sets.
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Return the training and test sets
    return X_train, X_test, y_train, y_test


# =====================================================================


def evaluate_classifier(X_train, X_test, y_train, y_test):
    '''
    Run multiple times with different classifiers to get an idea of the
    relative performance of each configuration.

    Returns a sequence of tuples containing:
        (title, precision, recall)
    for each learner.
    '''

    # Import some classifiers to test
    from sklearn.svm import LinearSVC, NuSVC
    from sklearn.ensemble import AdaBoostClassifier

    # We will calculate the P-R curve for each classifier
    from sklearn.metrics import precision_recall_curve, f1_score
    
    # Here we create classifiers with default parameters. These need
    # to be adjusted to obtain optimal performance on your data set.
    
    # Test the linear support vector classifier
    classifier = LinearSVC(C=1)
    # Fit the classifier
    classifier.fit(X_train, y_train)
    score = f1_score(y_test, classifier.predict(X_test))
    # Generate the P-R curve
    y_prob = classifier.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    # Include the score in the title
    yield 'Linear SVC (F1 score={:.3f})'.format(score), precision, recall

    # Test the Nu support vector classifier
    classifier = NuSVC(kernel='rbf', nu=0.5, gamma=1e-3)
    # Fit the classifier
    classifier.fit(X_train, y_train)
    score = f1_score(y_test, classifier.predict(X_test))
    # Generate the P-R curve
    y_prob = classifier.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    # Include the score in the title
    yield 'NuSVC (F1 score={:.3f})'.format(score), precision, recall

    # Test the Ada boost classifier
    classifier = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm='SAMME.R')
    # Fit the classifier
    classifier.fit(X_train, y_train)
    score = f1_score(y_test, classifier.predict(X_test))
    # Generate the P-R curve
    y_prob = classifier.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    # Include the score in the title
    yield 'Ada Boost (F1 score={:.3f})'.format(score), precision, recall

# =====================================================================


def plot(results):
    '''
    Create a plot comparing multiple learners.

    `results` is a list of tuples containing:
        (title, precision, recall)
    
    All the elements in results will be plotted.
    '''

    # Plot the precision-recall curves

    fig = plt.figure(figsize=(6, 6))
    fig.canvas.set_window_title('Classifying data from ' + URL)

    for label, precision, recall in results:
        plt.plot(recall, precision, label=label)

    plt.title('Precision-Recall Curves')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.legend(loc='lower left')

    # Let matplotlib improve the layout
    plt.tight_layout()

    # ==================================
    # Display the plot in interactive UI
    plt.show()

    # To save the plot to an image file, use savefig()
    #plt.savefig('plot.png')

    # Open the image file with the default image viewer
    #import subprocess
    #subprocess.Popen('plot.png', shell=True)

    # To save the plot to an image in memory, use BytesIO and savefig()
    # This can then be written to any stream-like object, such as a
    # file or HTTP response.
    #from io import BytesIO
    #img_stream = BytesIO()
    #plt.savefig(img_stream, fmt='png')
    #img_bytes = img_stream.getvalue()
    #print('Image is {} bytes - {!r}'.format(len(img_bytes), img_bytes[:8] + b'...'))

    # Closing the figure allows matplotlib to release the memory used.
    plt.close()


# =====================================================================


if __name__ == '__main__':
    # Download the data set from URL
    print(f'Downloading data from {URL}')
    # frame = download_data()

    # Process data into feature and label arrays
    # print("Processing {} samples with {} attributes".format(len(frame.index), len(frame.columns)))
    # X_train, X_test, y_train, y_test = get_features_and_labels(frame)

    # Evaluate multiple classifiers on the data
    print("Evaluating classifiers")
    # results = list(evaluate_classifier(X_train, X_test, y_train, y_test))

    # Display the results
    print("Plotting the results")
    # plot(results)
