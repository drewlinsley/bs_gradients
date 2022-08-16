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
import xarray as xr
import os


parent = '/cifs/data/tserre_lrs/projects/prj_brainscore/hackaton2021/.brainio/'
parent_imgs ='/cifs/data/tserre_lrs/projects/prj_brainscore/hackaton2021/.brainio/image_dicarlo_hvm-private/'
train_ids = 'data/train_ids.npy'
test_ids = 'data/test_ids.npy'
pvtdf = pd.read_csv(os.path.join(parent_imgs,'image_dicarlo_hvm-private.csv'))
ndat = xr.open_dataset(os.path.join(parent,'assy_dicarlo_MajajHong2015_private/assy_dicarlo_MajajHong2015_private.nc'))
ax = xr.open_dataset('masks_private.nc')

size = 224
batch_size = 64


meta={}
for s in models_meta : 
  if s['fields']['model'] not in meta: 
    meta[s['fields']['model']]={s['fields']['key']:s['fields']['value']}
  else: 
    meta[s['fields']['model']][s['fields']['key']]=s['fields']['value']


train_ids, test_ids = np.load(train_ids, allow_pickle=True), np.load(test_ids, allow_pickle=True)
all_ids = np.concatenate((train_ids,test_ids))
images_paths = [benchmark._assembly.stimulus_set.get_image(img_id) for img_id in all_ids]



public_masks = xr.open_dataset('masks_xr.nc')
csvf = '/cifs/data/tserre_lrs/projects/prj_brainscore/hackaton2021/.brainio/image_dicarlo_hvm-public/image_dicarlo_hvm-public.csv'
public = pd.read_csv(csvf)

preprocess = tf.keras.applications.vgg16.preprocess_input
train_ids, test_ids = np.load('data/train_ids.npy', allow_pickle=True), np.load('data/test_ids.npy', allow_pickle=True)
all_ids = np.array(list(train_ids)+list(test_ids))
X_all = load_ids(all_ids)
#X_all = preprocess(X_all)
X_train, X_test = load_ids(train_ids), load_ids(test_ids)
Y_all = np.concatenate((Y_train,Y_test))


dataset = []
for j,ids in enumerate(all_ids):
    #for i in range(5760):
    idx = np.where(public_masks.image_id==ids)[0][0]
    msk = public_masks.__xarray_dataarray_variable__[idx]
    filen = str(msk.image_file_name.values)
    img_id = str(msk.image_id.values)
    if ids == img_id: 
        category = public[public['image_id']==img_id]['category_name'].values[0]
        obj_name = public[public['image_id']==img_id]['object_name'].values[0]
        msk = np.array(msk).astype(np.int)
        #plt.imshow(msk)
        x = np.argmax(msk,axis=0)
        cx = np.max(x[x>0])*0.5 + np.min(x[x>0])*0.5
        
        y = np.argmax(msk,axis=1)
        cy = np.max(y[y>0])*0.5 + np.min(y[y>0])*0.5 
        
        corner = [x- np.max(x[x>0]),y-np.max(y)]
        dataset.append([filen,img_id,category,obj_name,cx,cy,corner,Y_all[j],X_all[j],msk])

public[public['image_id']==img_id]['category_name'].values
ds = pd.DataFrame(dataset,columns=['file_name','image_id','category','object','x','y','corner','recording','image','msk'])
ds[['file_name','image_id','category','x','y']].to_csv('images_with_xy.csv')

model = tf.keras.applications.vgg16.VGG16(weights="imagenet")
preprocess = tf.keras.applications.vgg16.preprocess_input


# BASELINE TEST 
leng = X_all.shape[0]
X_train = X_all[:int(leng*0.9)]
X_test =   X_all[int(leng*0.9):]
Y_train = Y_all[:int(leng*0.9)]
Y_test = Y_all[int(leng*0.9):]
X_train_preprocessed = np.array(preprocess(np.array(X_train, copy=True)), np.float16)
X_test_preprocess = np.array(preprocess(np.array(X_test, copy=True)), np.float16)


layer = 'block4_pool'
activation_model = tf.keras.Model(model.input, model.get_layer(layer).output)
brain_score(activation_model, X_train, Y_train, X_test, Y_test)


