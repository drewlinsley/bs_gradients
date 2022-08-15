import functools
import tensorflow as tf
import cv2
from model_tools.activations.pytorch import load_preprocess_images
from model_tools.activations.pytorch import PytorchWrapper
from model_tools.activations.keras import KerasWrapper
from glob import glob
from tf_utils import Tensorflow2Wrapper
import efficientnet 
from brainscore import score_model
import pandas as pd
from model_tools.brain_transformation import ModelCommitment


_mean_imagenet = tf.constant([0.485, 0.456, 0.406], shape=[1, 1, 3], dtype=tf.float32)
_std_imagenet =  tf.constant([0.229, 0.224, 0.225], shape=[1, 1, 3], dtype=tf.float32)


def load_images(image_filepaths,image_size):
    return np.array([load_image(image_filepath,image_size) for image_filepath in image_filepaths])

def load_image(image_filepath,image_size):
    
    original_image = cv2.imread(image_filepath)
    height, width = original_image.shape[:2]
    
    if len(original_image.shape)==2:
        original_image = gray2rgb(original_image)
    #image = transform_gen.get_transform(original_image).apply_image(original_image)
    
    image = tf.image.resize(original_image,(image_size,image_size)).numpy()
    image = tf.cast(image, tf.float32)/255.0
    image -= _mean_imagenet
    image /= _std_imagenet
    
    #inputs = {"image": image, "height": height, "width": width}
        
    return image

def load_preprocess_images(image_filepaths, image_size=256,**kwargs):
    #torch.cuda.empty_cache()
    images = load_images(image_filepaths,image_size)
    return images



preprocessing = functools.partial(load_preprocess_images, image_size=224)
efficientnet.init_tfkeras_custom_objects()
# models_eff = glob('/cifs/data/tserre_lrs/projects/prj_metapredictor/meta_models/models/eff*h5')
models_eff=['/cifs/data/tserre_lrs/projects/prj_metapredictor/meta_models/models/efficientnet_baseline.h5',
 '/cifs/data/tserre_lrs/projects/prj_metapredictor/meta_models/models/efficientnet_classic-dream-12.h5',
 '/cifs/data/tserre_lrs/projects/prj_metapredictor/meta_models/models/efficientnet_gallant-dust_1.h5']
models =[
    '/cifs/data/tserre_lrs/projects/prj_metapredictor/meta_models/models/vgg_baseline.h5',
    '/cifs/data/tserre_lrs/projects/prj_metapredictor/meta_models/models/vgg_frosty_eon.h5',
    '/cifs/data/tserre_lrs/projects/prj_metapredictor/meta_models/models/resnet50_baseline.h5',
    '/cifs/data/tserre_lrs/projects/prj_metapredictor/meta_models/models/saliency_volcanic_monkey.h5',
    '/cifs/data/tserre_lrs/projects/prj_metapredictor/meta_models/models/vgg_silver_moon.h5',
]

results = []
for MODEL_NAME in models_eff:
    print(MODEL_NAME)
    model = tf.keras.models.load_model(MODEL_NAME,compile=False)
    layers = [n.name for n in model.layers[-10:]]
    activations_model = Tensorflow2Wrapper(identifier=MODEL_NAME, model=model, preprocessing=preprocessing)
    model = ModelCommitment(identifier=MODEL_NAME, activations_model=activations_model,layers = layers )
    score_v4 = score_model(model_identifier=MODEL_NAME, model=model,
                    benchmark_identifier='dicarlo.MajajHong2015public.V4-pls')
    print(score_v4)
    score_it = score_model(model_identifier=MODEL_NAME, model=model,
                    benchmark_identifier='dicarlo.MajajHong2015public.IT-pls',verbose=0)
    print(score_it)
    
    results.append([MODEL_NAME,score_it,score_v4])
    print(results)
    rdf = pd.DataFrame(results,columns=['model','score_it','score_v4'])
    rdf.to_csv('bs_score_our_models_eff_2.csv')

parse = []
for i in range(len(rdf)):
    row =  rdf.iloc[i]
    model = row['model'].split('/')[-1]
    raw_it_m = row['score_it'].raw.values[0]
    raw_it_s = row['score_it'].raw.values[1]
    raw_v4_m = row['score_v4'].raw.values[0]
    raw_v4_s = row['score_v4'].raw.values[1]
    ceiling_it_m = row['score_it'].ceiling.values[0]
    ceiling_it_s = row['score_it'].ceiling.values[1]
    ceiling_v4_m = row['score_v4'].ceiling.values[0]
    ceiling_v4_s = row['score_v4'].ceiling.values[1]
    parse.append([model,raw_it_m,raw_it_s,raw_v4_m,raw_v4_s,ceiling_it_m,ceiling_it_s,ceiling_v4_m,ceiling_v4_s])

parsedf_eff = pd.DataFrame(
    parse,
    columns=[
        'model',
        'raw_it_m',
        'raw_it_s',
        'raw_v4_m',
        'raw_v4_s',
        'ceiling_it_m',
        'ceiling_it_s',
        'ceiling_v4_m',
        'ceiling_v4_s'
    ]
)

model = ModelCommitment(
    identifier='my-model',
    activations_model=activations_model,
    layers=['conv1', 'relu1', 'relu2'])

# The score_model will score the model on the specified benchmark.
# When the model is asked to output activations for the IT region, it will first search for the best layer
# and then only output this layer's activations.
score = score_model(
    model_identifier=model.identifier,
    model=model,
    benchmark_identifier='dicarlo.MajajHong2015public.IT-pls')
print(score)
