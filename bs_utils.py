import cv2
import numpy as np
import tensorflow as tf
from sklearn.cross_decomposition import PLSRegression
from brainscore.benchmarks.public_benchmarks import MajajHongITPublicBenchmark


benchmark = MajajHongITPublicBenchmark()


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


