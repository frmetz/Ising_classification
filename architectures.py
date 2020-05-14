########################################
# copied and modified from: https://github.com/google/jax/issues/1393
########################################


from jax.config import config
#config.update("jax_enable_x64", True)
from jax import jit, device_put
#from jax.experimental.stax import relu, BatchNorm

import functools
import itertools
from jax import lax
from jax import random
import jax.numpy as jnp
import numpy as np

from jax.nn import (relu, log_softmax, softmax, softplus, sigmoid, elu,
                    leaky_relu, selu, gelu, normalize)
from jax.nn.initializers import glorot_normal, normal, ones, zeros

# aliases for backwards compatibility
# glorot = glorot_normal
# randn = normal
# logsoftmax = log_softmax




def periodic_padding(inputs,filter_shape,strides):
    n_x=filter_shape[0]-strides[0]
    n_y=filter_shape[1]-strides[1]
    return jnp.pad(inputs, ((0,0),(0,0),(0,n_x),(0,n_y)), mode='wrap')


def PeriodicConv(out_chan, filter_shape,
                strides=None, padding='VALID', dimension_numbers = ('NHWC', 'HWIO', 'NHWC'), W_init=None,
                b_init=normal(1e-6), ignore_b=False, dtype=jnp.float64):
    """Layer construction function for a general convolution layer."""
    lhs_spec, rhs_spec, out_spec = dimension_numbers
    one = (1,) * len(filter_shape)
    strides = strides or one
    W_init = W_init or glorot_normal(rhs_spec.index('I'), rhs_spec.index('O'))

    def init_fun(rng, input_shape):

        # add padding dimensions for periodic BC; move this line into conv_general_shape_tuple after defining padding='PERIODIC'
        input_shape+=np.array((0,0)+strides)

        filter_shape_iter = iter(filter_shape)
        kernel_shape = [out_chan if c == 'O' else
                        input_shape[lhs_spec.index('C')] if c == 'I' else
                        next(filter_shape_iter) for c in rhs_spec]
        output_shape = lax.conv_general_shape_tuple(
            input_shape, kernel_shape, strides, padding, dimension_numbers)

        k1, k2 = random.split(rng)

        if not ignore_b:
            bias_shape = [out_chan if c == 'C' else 1 for c in out_spec]
            bias_shape = tuple(itertools.dropwhile(lambda x: x == 1, bias_shape))

            W, b = W_init(k1, kernel_shape, dtype=dtype), b_init(k2, bias_shape, dtype=dtype)
            return output_shape, (W, b)
        else:
            W = W_init(k1, kernel_shape, dtype=dtype)
            return output_shape, (W, )

    def apply_fun(params, inputs, **kwargs):

        # move this line into lax.conv_general_dilated after defining padding='PERIODIC'
        inputs=periodic_padding(inputs.astype(dtype),filter_shape,strides)

        if not ignore_b:
            W, b = params
            return lax.conv_general_dilated(inputs, W, strides, padding, one, one,
                                            dimension_numbers) + b
        else:
            W = params
            return lax.conv_general_dilated(inputs, W, strides, padding, one, one,
                                            dimension_numbers)

    return init_fun, apply_fun
