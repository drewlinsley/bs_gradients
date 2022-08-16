import tensorflow as tf
import models_meta
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import tensorflow as tf
from tensorflow.keras.layers import *
import brainscore
import pandas as pd
import bs_utils



size = 224

meta={}
for s in models_meta : 
  if s['fields']['model'] not in meta: 
    meta[s['fields']['model']]={s['fields']['key']:s['fields']['value']}
  else: 
    meta[s['fields']['model']][s['fields']['key']]=s['fields']['value']








batch_size = 64



