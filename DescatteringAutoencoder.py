import numpy as np

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv1D, AveragePooling1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Lambda, Conv2DTranspose, Activation, Reshape
from tensorflow.keras.regularizers import l2, l1_l2
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential

def Descattering_Autoencoder(input_spectrum,
                             filters=[32, 64, 32, 16, 8, 4, 16, 32, 64, 32, 1], 
                             kernel_sizes=[23,17,13,7, 5, 3, 3, 5, 7, 11, 17],
                             strides=[1, 2, 1, 1, 1, 2, 2, 2, 4, 2, 1],
                             l2_reg = 0.0,
                             l1_reg = 0.0,
                             pooling="average"):

    Pooling = AveragePooling1D if pooling == "average" else MaxPooling1D
    bn_idx = np.argmin(np.array(filters[:-1]))
    
    # Encoder
    encoder = input_spectrum
    k, j = 1, 1
    for flter, krnl_size, strd in zip(filters[:bn_idx], kernel_sizes[:bn_idx], strides[:bn_idx]):
        encoder = Conv1D(filters=flter, 
                         kernel_size=krnl_size,
                         strides=strd, 
                         padding='same', 
                         kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg), 
                         bias_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                         name=f"Conv_{k}")(encoder) 

        encoder = BatchNormalization(name=f"BatchNorm_{k}")(encoder)
        encoder = Activation("relu", name=f"ReLU_{k}")(encoder)
        encoder = Pooling(pool_size=2, name=f"Pooling_{k}")(encoder)
        k += 1
        
    # Decoder
    decoder = encoder
    for flter, krnl_size, strd in zip(filters[bn_idx:], kernel_sizes[bn_idx:], strides[bn_idx:]):
        decoder = Conv1DTranspose(filters=flter, 
                                  kernel_size=krnl_size,
                                  strides=strd, 
                                  padding='same',
                                  kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg), 
                                  bias_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                                  name=f"TranspConv_{j}")(decoder)
            
        decoder = BatchNormalization(name=f"Batch_Norm_{k}")(decoder)
        decoder = Activation("relu", name=f"ReLU_{k}")(decoder)
        k += 1
        j += 1
        
    output_spectrum = Conv1D(filters=filters[-1], 
                            kernel_size=kernel_sizes[-1], 
                            strides=strides[-1], 
                            padding='same', 
                            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg), 
                            bias_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                            name=f"Conv_{k-j+1}")(decoder) 
    
    return output_spectrum


class Conv1DTranspose(Layer):
    def __init__(self, filters, kernel_size, strides, padding, 
                       kernel_regularizer, bias_regularizer, *args, **kwargs):
        self._filters = filters
        self._kernel_size = (1, kernel_size)
        self._strides = (1, strides)
        self._padding = padding
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._args, self._kwargs = args, kwargs
        super(Conv1DTranspose, self).__init__(**kwargs)

    def build(self, input_shape):
        self._model = Sequential(name=self.name)
        self._model.add(Lambda(lambda x: K.expand_dims(x,axis=1), batch_input_shape=input_shape))
        self._model.add(Conv2DTranspose(filters=self._filters,
                                        kernel_size=self._kernel_size,
                                        strides=self._strides,
                                        padding=self._padding,
                                        kernel_regularizer=self._kernel_regularizer,
                                        bias_regularizer=self._bias_regularizer,  
                                        *self._args, **self._kwargs))
        self._model.add(Lambda(lambda x: x[:,0]))
        super(Conv1DTranspose, self).build(input_shape)

    def call(self, x):
        return self._model(x)

    def compute_output_shape(self, input_shape):
        return self._model.compute_output_shape(input_shape)