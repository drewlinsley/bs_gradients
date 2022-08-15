from constants import *
import tensorflow as tf
#import cv2
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
import cv2
import tensorflow as tf
from tensorflow.keras.layers import *
import brainscore
import pandas as pd

from brainscore.benchmarks.public_benchmarks import MajajHongITPublicBenchmark
import cv2
from tensorflow.keras.applications import *
import inspect
import re
import xarray as xr
def matching_strings(pattern, text): 
    left = 0
    count = 0
    while True:
        match = re.search(pattern, text[left:])
        if not match:
            break
        count += 1
        left += match.start() + 1
   
    return count 
size = 224



benchmark = MajajHongITPublicBenchmark()

def load_ids(ids):
  images_paths = [benchmark._assembly.stimulus_set.get_image(img_id) for img_id in ids]
  images = np.array([cv2.imread(p) for p in images_paths])
  images = tf.image.resize(images, (size, size)).numpy()
  return images



def set_size(w,h):
  plt.rcParams["figure.figsize"] = [w,h]

def show(img, p=False, **kwargs):
  """manages to display an image"""
  img = np.array(img, dtype=np.float32)
  # check if channel first
  if img.shape[0] == 1:
    img = img[0]
  elif img.shape[0] == 3:
    img = np.moveaxis(img, 0, 2)
  # check if cmap
  if img.shape[-1] == 1:
    img = img[:,:,0]
  # normalize
  if img.max() > 1 or img.min() < 0:
    img -= img.min(); img/=img.max()

  plt.imshow(img, **kwargs)
  plt.axis('off')
  plt.grid(None)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


meta={}
for s in MODELS_META : 
  if s['fields']['model'] not in meta: 
    meta[s['fields']['model']]={s['fields']['key']:s['fields']['value']}
  else: 
    meta[s['fields']['model']][s['fields']['key']]=s['fields']['value']
    
batch_size = 64

def batch_predict(model, inputs, batch_size=batch_size):
  activations = []
  
  for batch_x in tf.data.Dataset.from_tensor_slices(inputs).batch(batch_size):
    #print('here')
    batch_act = model(batch_x)
    batch_act = np.array(batch_act, np.float16)
    activations += list(batch_act)

  return np.array(activations, np.float16)

def tf_pearson_loss(x, y, axis=-1):
  """
  Compute the Pearson's correlation as the centered cosine similarity
  """
  x_means = tf.reduce_mean(x, axis, keepdims=True)
  y_means = tf.reduce_mean(y, axis, keepdims=True)

  x_centered = x - x_means
  y_centered = y - y_means

  inner_products = tf.reduce_sum(x_centered * y_centered, axis)
  norms = tf.sqrt(tf.reduce_sum(x_centered**2, axis)) * \
         tf.sqrt(tf.reduce_sum(y_centered**2, axis))

  correlations = inner_products / norms
  
  return correlations

def brain_score(activation_model, X_train, Y_train, X_test, Y_test):
  A_train = batch_predict(activation_model, X_train, batch_size)
  A_train = A_train.reshape((len(X_train), -1))

  pls_reg = PLSRegression(20, scale=False, tol=1e-4)
  pls_reg.fit(A_train, Y_train)

  pls_kernel = tf.cast(pls_reg.coef_, tf.float16)
  Y_test = tf.cast(Y_test, tf.float16)

  A_test = batch_predict(activation_model, X_test, batch_size)
  A_test = A_test.reshape((len(X_test), -1))

  Y_pred = tf.matmul(A_test, pls_kernel)

  correlations = tf_pearson_loss(
      tf.transpose(Y_pred),
      tf.transpose(Y_test)
  )

  score = np.mean(correlations)
  print('Brainscore:',score)
  print('mean:',np.median(correlations)) 
  print('std:',np.std(correlations))
  return score, pls_kernel,correlations




# Closest different Category

def baseline_closest_different_category_test(ds):
    x = np.array(ds.x.tolist())
    y = np.array(ds.y.tolist())
    distance_matrix = np.sqrt(np.square(x - x.reshape(-1,1)) + np.square(y - y.reshape(-1,1)))
    new_dataset =[]
    
    for i in range(len(ds)):
        sample = ds.iloc[i]
        category1 = sample['category']
        for j in range(1,len(ds)):
            closest_id = np.argsort(distance_matrix[i])[j]
            sample2 = ds.iloc[closest_id]
            category2 = sample2['category']
            if category1 != category2: 
                new_dataset.append(sample2.values)
                break 
    
    return new_dataset

def closest_different_category_test(ds):
    x = np.array(ds.x.tolist())
    y = np.array(ds.y.tolist())
    distance_matrix = np.sqrt(np.square(x - x.reshape(-1,1)) + np.square(y - y.reshape(-1,1)))
    new_dataset =[]
    
    for i in range(len(ds)):
        sample = ds.iloc[i]
        vals1 = sample.values
        category1 = sample['category']
        for j in range(1,len(ds)):
            closest_id = np.argsort(distance_matrix[i])[j]
            sample2 = ds.iloc[closest_id]
            category2 = sample2['category']
            vals2 = sample2.values
            vals2[-2]= vals1[-2]
            if category1 != category2: 
                new_dataset.append(vals2)
                break 
    
    return new_dataset

def box_inthemiddle_different_category_test(ds):
    x = np.array(ds.x.tolist())
    y = np.array(ds.y.tolist())
    distance_matrix = np.sqrt(np.square(x - x.reshape(-1,1)) + np.square(y - y.reshape(-1,1)))
    new_dataset =[]
    
    for i in range(len(ds)):
        sample = ds.iloc[i]
        vals1 = sample.values
        category1 = sample['category']
        for j in range(1,len(ds)):
            closest_id = np.argsort(distance_matrix[i])[j]
            sample2 = ds.iloc[closest_id]
            category2 = sample2['category']
            vals2 = sample2.values
            vals2[-2]= vals1[-2]
            if category1 != category2: 
                new_dataset.append(vals2)
                break 
    
    return new_dataset

def per_category_brainscore(ds,activation_model,model_identifier):
    
    categories = ds.category.unique()
    results=[]
    
    for c in categories:
        X = np.array(ds[ds.category==c].image.tolist())
        Y = np.array(ds[ds.category==c].recording.tolist())
        
        leng = X.shape[0]
        X_train = X[:int(leng*0.9)]
        X_test =   X[int(leng*0.9):]
        Y_train = Y[:int(leng*0.9)]
        Y_test = Y[int(leng*0.9):]
        X_train_preprocessed = np.array(preprocess(np.array(X_train, copy=True)), np.float16)
        X_test_preprocess = np.array(preprocess(np.array(X_test, copy=True)), np.float16)
        score,_,correlations = brain_score(activation_model, X_train, Y_train, X_test, Y_test)
        results.append([c,score,np.median(correlations),np.std(correlations),np.max(correlations),np.min(correlations)])
        print(f'{model_identifier} at category {c} mean score {score} std {np.std(correlations)}')
    rf = pd.DataFrame(results,columns=['category','mean score','median score','std','max','min'])
    rf.to_csv(f'{model_identifier}_per_category_bs.csv')
    return rf 

def per_object_brainscore(ds,activation_model,model_identifier):
    
    categories = ds.object.unique()
    results=[]
    
    for c in categories:
        X = np.array(ds[ds.object==c].image.tolist())
        Y = np.array(ds[ds.object==c].recording.tolist())
        
        leng = X.shape[0]
        X_train = X[:int(leng*0.9)]
        X_test =   X[int(leng*0.9):]
        Y_train = Y[:int(leng*0.9)]
        Y_test = Y[int(leng*0.9):]
        X_train_preprocessed = np.array(preprocess(np.array(X_train, copy=True)), np.float16)
        X_test_preprocess = np.array(preprocess(np.array(X_test, copy=True)), np.float16)
        score,_,correlations = brain_score(activation_model, X_train, Y_train, X_test, Y_test)
        results.append([c,score,np.median(correlations),np.std(correlations),np.max(correlations),np.min(correlations)])
        print(f'{model_identifier} at category {c} mean score {score} std {np.std(correlations)}')
    rf = pd.DataFrame(results,columns=['category','mean score','median score','std','max','min'])
    rf.to_csv(f'{model_identifier}_per_category_bs.csv')
    return rf 

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# define example
data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']

def one_hot(data):
    values = np.array(data)

    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    #print(onehot_encoded)
    return onehot_encoded

def countPairs(s1, s2) :
    n1 =len(s1)-1
    n2 =len(s2)-1
    # To store the frequencies of characters
    # of string s1 and s2
    freq1 = [0] * 36;
    freq2 = [0] * 36;
 
    # To store the count of valid pairs
    count = 0;
 
    # Update the frequencies of
    # the characters of string s1
    for i in range(n1) :
        freq1[ord(s1[i]) - ord('a')] += 1;
 
    # Update the frequencies of
    # the characters of string s2
    for i in range(n2) :
        freq2[ord(s2[i]) - ord('a')] += 1;
 
    # Find the count of valid pairs
    for i in range(26) :
        count += min(freq1[i], freq2[i]);
 
    return count;


def fine_tune_model(model,train_ds,test_ds,layer=None):
    #model.summary()
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    #x = model(inputs)
    if layer:
        for l in model.layers:
            #print(l.name,layer,l.name==layer)
            if l.name != layer:
                l.trainable=True
            else:
                print(l.name)
                l.trainable =True
    x = model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # A Dense classifier with a single unit (binary classification)
    outputs = tf.keras.layers.Dense(256)(x)
    outputs = tf.keras.layers.Dense(8,activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),loss='categorical_crossentropy',metrics='accuracy')
    model.fit(train_ds, epochs=25,validation_data=test_ds)
    return model


public_masks = xr.open_dataset('masks_xr.nc')
csvf = '/cifs/data/tserre_lrs/projects/prj_brainscore/hackaton2021/.brainio/image_dicarlo_hvm-public/image_dicarlo_hvm-public.csv'
public = pd.read_csv(csvf)

preprocess = tf.keras.applications.vgg16.preprocess_input
train_ids, test_ids = np.load('data/train_ids.npy', allow_pickle=True), np.load('data/test_ids.npy', allow_pickle=True)
all_ids = np.array(list(train_ids)+list(test_ids))
X_all = load_ids(all_ids)
#X_all = preprocess(X_all)
Y_train, Y_test = np.load('data/y_train.npy'), np.load('data/y_test.npy')
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
        
ds = pd.DataFrame(dataset,columns=['file_name','image_id','category','object','x','y','corner','recording','image','msk'])
ds[['file_name','image_id','category','x','y']].to_csv('images_with_xy.csv')


X = np.array(ds.image.tolist())
y = one_hot(ds.category.tolist())
Y = np.array(ds.recording.tolist())

X_train, X_test, y_train_cat, y_test_cat = train_test_split(X, y, test_size=0.1, random_state=42,stratify=y)
X_train_bs, X_test_bs, y_train_bs, y_test_bs = train_test_split(X, Y, test_size=0.1, random_state=42)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train_cat))
# need to repeat the dataset for epoch number of times, as all the data needs
# to be fed to the dataset at once
train_dataset = train_dataset.shuffle(1000).batch(32)

test_dataset =tf.data.Dataset.from_tensor_slices((X_test, y_test_cat))
test_dataset = test_dataset.batch(16)

keras_models= inspect.getmembers(tf.keras.applications)
ft_bs =[ ]
fine_tuned = pd.read_csv('fine_tunned_all_layer_trained_scratch.csv')
for c,km in enumerate(keras_models):
    print(c)
    if type(km[1])== type(tf.keras.applications.resnet_v2.ResNet50V2):
        if km[0] in fine_tuned.model.unique():
            continue
        pattern = km[0].lower()
        text = []
        found=0
        for k in meta.keys():
            
            count = matching_strings(pattern,k.replace('-','').lower())
            #count = countPairs(pattern,k.replace('-','').lower())
            
            if count>0:
                print('Possible_match found:',pattern, k.replace('-',''), count)
                found=1
                for layer_type in meta[k]: 
                    if layer_type !='IT_layer': continue
                    layer = meta[k][layer_type]
                    print(layer)
                    #print(f'data/{k}_{layer_type}_{layer}_features.npy')
                    
                    model = km[1].__call__(weights=None,include_top=False)
#                   model.summary()
                    try:
                        activation_model = tf.keras.Model(model.input, model.get_layer(layer).output)
                    except: 
                        found=0
                        print('not layer like that')
                        continue
                    snf,_,_ = brain_score(activation_model, X_train_bs, y_train_bs, X_test_bs, y_test_bs)
                    print(pattern)
                    model = fine_tune_model(model,train_dataset,test_dataset,layer)
                    model = model.get_layer(pattern)
                    activation_model = tf.keras.Model(model.input, model.get_layer(layer).output)
                    A = batch_predict(activation_model, X_all, batch_size)
                    A = A.reshape((len(X_all), -1))
                    file_name = f'data/finetuned_{k}_{layer_type}_{layer}_features_scratch.npy'
                    np.save(file_name,A)
                    print(file_name)
                    s,_,_ = brain_score(activation_model, X_train_bs, y_train_bs, X_test_bs, y_test_bs)
                    ft_bs.append([km[0],s,snf,layer])
                    print(f'Before training {snf} and after training {s} layer {layer} model {pattern}')
        
            
        
        
        
        
        
        if found==0:
            #print(pattern==km[0].lower(),matching_strings_strings())
            print('MODEL NOT FOUND',pattern,km[0].lower(),meta[k]['IT_layer'])
            count = matching_strings(pattern,k.replace('-','').lower())
            model = km[1].__call__(weights=None,include_top=False)
            layer_type='it'
            for l in model.layers[-4:-1]:
                layer = l.name
                activation_model = tf.keras.Model(model.input, model.get_layer(layer).output)
                try:
                    snf,_,_ = brain_score(activation_model, X_train_bs, y_train_bs, X_test_bs, y_test_bs)
                except: 
                    print('problem with dim')
                    continue
                pattern = model.name
                print(pattern)
                model = fine_tune_model(model,train_dataset,test_dataset,layer)
                model = model.get_layer(pattern)
                activation_model = tf.keras.Model(model.input, model.get_layer(layer).output)
                A = batch_predict(activation_model, X_all, batch_size)
                A = A.reshape((len(X_all), -1))
                file_name = f'data/finetuned_{pattern}_{layer_type}_{layer}_features_scratch.npy'
                np.save(file_name,A)
                print(file_name)
                s,_,_ = brain_score(activation_model, X_train_bs, y_train_bs, X_test_bs, y_test_bs)
                ft_bs.append([km[0],s,snf,layer])
                print(f'Before training {snf} and after training {s} layer {layer} model {pattern}')

                fine_tunned = pd.DataFrame(ft_bs,columns=['model','after training','before training', 'layer'])
                fine_tunned.to_csv('fine_tunned_all_layer_trained_scratch.csv')
