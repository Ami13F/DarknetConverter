from model import YoloModel
import numpy as np
import os.path
from utils import choose_model


class WeightsConvertor():
    def __init__(self, model, weights_file):
        self.__block = model.blocks
        self.model = model.create_network()
        self.weights_file = weights_file

    def load_weights(self):
        with open(self.weights_file, 'rb') as fp:
            # skip header info
            np.fromfile(fp, dtype=np.int32, count=5)

            blocks = self.__block

            for i, block in enumerate(blocks[1:]):
                if block["type"] == "convolutional":
                    conv_layer = self.model.get_layer("conv-" + str(i))

                    filters = conv_layer.filters
                    k_size = conv_layer.kernel_size[0]
                    input_dim = conv_layer.input_shape[-1]

                    if "batch_normalize" in block:
                        norm_layer = self.model.get_layer(
                            "batch_norm-" + str(i))
                        # get 4 params
                        bn_weights = np.fromfile(
                            fp, dtype=np.float32, count=4 * filters)
                        # read beta, gamma, mean, variance
                        bn_weights = bn_weights.reshape(
                            (4, filters))[[1, 0, 2, 3]]
                    else:
                        conv_bias = np.fromfile(
                            fp, dtype=np.float32, count=filters)

                    # darknet shape (out_dim, in_dim, height, width)
                    conv_shape = (filters, input_dim, k_size, k_size)
                    conv_weights = np.fromfile(
                        fp, dtype=np.float32, count=np.product(conv_shape))
                    conv_weights = conv_weights.reshape(
                        conv_shape).transpose([2, 3, 1, 0])

                    if "batch_normalize" in block:
                        norm_layer.set_weights(bn_weights)
                        conv_layer.set_weights([conv_weights])
                    else:
                        conv_layer.set_weights([conv_weights, conv_bias])
