#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import itertools

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras import optimizers
from keras import initializers

import keras.utils

#%%
