import numpy as np
import tensorflow.keras
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer # corrected import
import tensorflow as tf
from tensorflow.keras.layers import Conv1D # corrected import

model_dir = "models/"
generation_dir = 'data/generated_data/'

generation_number = 10000

n = 32
latent_dim = 64
hidden_dim = 64


def rebuild_model(model_path):

    class GCNN(Layer):
        def __init__(self, output_dim=None, residual=False, **kwargs):
            super(GCNN, self).__init__(**kwargs)
            self.output_dim = output_dim
            self.residual = residual
            self.conv1d = None # Initialize conv1d layer

        def build(self, input_shape):
            if self.output_dim == None:
                self.output_dim = input_shape[-1]
            # Use Conv1D Layer
            self.conv1d = Conv1D(
                filters=self.output_dim * 2,
                kernel_size=3,
                padding='same',
                kernel_initializer='glorot_uniform',
                name='gcnn_conv1d'
            )
            self.built = True

        def call(self, x):
            _ = self.conv1d(x)
            print("input", x)
            print("conv", _)
            print("output_dim", self.output_dim)
            _ = _[:, :, :self.output_dim] * tf.nn.sigmoid(_[:, :, self.output_dim:]) # Use tf.nn.sigmoid
            print("output", _)
            if self.residual:
                return _ + x
            else:
                return _

    input_sentence = Input(shape=(n,), dtype='int32')
    input_vec = Embedding(16, hidden_dim)(input_sentence)
    h = GCNN(residual=True)(input_vec)
    h = GCNN(residual=True)(h)
    h = GlobalAveragePooling1D()(h)

    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0, stddev=1)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = Lambda(sampling)([z_mean, z_log_var])

    decoder_hidden = Dense(hidden_dim * n)
    decoder_cnn = GCNN(residual=True)
    decoder_dense = Dense(16, activation='softmax')

    h = decoder_hidden(z)
    h = Reshape((n, hidden_dim))(h)
    h = decoder_cnn(h)
    output = decoder_dense(h)

    vae = Model(input_sentence, output)
    vae.load_weights(model_path)

    decoder_input = Input(shape=(latent_dim,))
    _ = decoder_hidden(decoder_input)
    _ = Reshape((n, hidden_dim))(_)
    _ = decoder_cnn(_)
    _output = decoder_dense(_)
    generator = Model(decoder_input, _output)

    return generator


def generation_store(target_generation, generation_path):
    f = open(generation_path, 'w', encoding='utf-8')
    f.writelines(target_generation)
    f.close()


def gen():
    r = generator.predict(np.random.randn(1, latent_dim))[0]
    r = r.argmax(axis=1)
    return r


if __name__ == "__main__":

    model_list = [
        "gcnn_vae.model.weights.h5"
        # "fixed_iid_100.model",
        # "low_64bit_subnet_100.model",
        # "slaac_eui64_100.model",
        # "slaac_privacy_100.model"
    ]

    generation_filename_list = [
        "6gcvae_generation_test.txt"
        # 'gcnn_vae_generation_fixed_iid_addresses_100_1M.txt',
        # 'gcnn_vae_generation_low_64bit_subnet_addresses_100_1M.txt',
        # 'gcnn_vae_generation_slaac_eui64_addresses_100_1M.txt',
        # 'gcnn_vae_generation_slaac_privacy_addresses_100_1M.txt'
    ]

    for model, generation_filename in zip(model_list, generation_filename_list):
        model_path = model_dir + model
        generation_path = generation_dir + generation_filename
        generator = rebuild_model(model_path)

        target_generation = []
        for i in range(generation_number):
            r = gen()
            gen_address = ""
            count = 0
            gen_address_list = [str(hex(i))[-1] for i in r]
            for i in gen_address_list:
                count += 1
                gen_address += i
                if count % 4 == 0:
                    gen_address += ":"
            gen_address = gen_address[:-1]
            target_generation.append(gen_address + '\n')
            # print(gen_address)
        generation_store(target_generation, generation_path)
