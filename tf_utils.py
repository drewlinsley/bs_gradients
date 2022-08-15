from model_tools.activations.core import ActivationsExtractorHelper
import numpy as np
from collections import OrderedDict


class Tensorflow2Wrapper:
    def __init__(self, identifier,model,preprocessing, *args, **kwargs):
        import tensorflow as tf
        self._model = model
        
        self._extractor = ActivationsExtractorHelper(identifier=identifier, get_activations=self.get_activations,
                                                     preprocessing=preprocessing, *args, **kwargs)
        self._extractor.insert_attrs(self)

    @property
    def identifier(self):
        return self._extractor.identifier

    @identifier.setter
    def identifier(self, value):
        self._extractor.identifier = value

    def __call__(self, *args, **kwargs):  # cannot assign __call__ as attribute due to Python convention
        return self._extractor(*args, **kwargs)
    def batch_predict(self, inputs, batch_size=10):
        activations = []
        for batch_x in tf.data.Dataset.from_tensor_slices(inputs).batch(batch_size):
            #print('here')
            batch_act = self._model(batch_x)
            batch_act = np.array(batch_act, np.float16)
            activations += list(batch_act)
        return np.array(activations, np.float16)
    
    def get_activations(self, images, layer_names):
        layer_outputs = []
        for layer in layer_names: 
            activation_model = tf.keras.Model(self._model.input, self._model.get_layer(layer).output)
            #import pdb;pdb.set_trace()
            layer_outputs.append(self.batch_predict(images))  # 0 to signal testing phase
        return OrderedDict([(layer_name, layer_output) for layer_name, layer_output in zip(layer_names, layer_outputs)])

