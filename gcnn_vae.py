import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
#from keras.engine.topology import Layer # replaced
from tensorflow.keras.layers import Layer # corrected import
from tensorflow.keras.layers import Conv1D # corrected import
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
import tensorflow as tf  # Ensure tensorflow is imported

dataset_path = 'data/processed_data/data.txt'
generated_path = 'data/generated_data/6gcvae_generation.txt'
# dataset_path = 'data/processed_data/slaac_privacy_addresses_gasser_data.txt'
# generated_path = 'data/generated_data/6vae_generation_slaac_privacy_addresses.txt'

n = 32
latent_dim = 64
hidden_dim = 64


def load_data(filename):

    f = open(filename, 'r', encoding='utf-8')
    raw_data = f.readlines()[:1000000]
    f.close()

    # 去除末尾换行符
    for i in range(len(raw_data)):
        raw_data[i] = raw_data[i][:-1]

    # 提取地址字符
    word_data = []
    for address in raw_data:
        address_data = []
        for i in range(len(address)):
            address_data.append(address[i])
        word_data.append(address_data)

    # 将地址字符转换为id
    v6dict = {
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
        '8': 8, '9': 9, 'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15
    }
    data = []
    for address in word_data:
        address_data = []
        for bit in address:
            address_data.append(v6dict[bit])
        data.append(address_data)

    target = np.ones(len(raw_data))
    x_train, x_test, y_train, y_test = train_test_split(data,
                                                        target,
                                                        test_size=0.2,
                                                        random_state=0)
    return x_train, x_test, y_train, y_test


def run_model():

    x_train, x_test, y_train, y_test = load_data(dataset_path)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    # x_train = x_train.astype('float32') / 15.
    # x_test = x_test.astype('float32') / 15.
    # 显式地将 x_train 和 x_test 转换为 int32 类型
    # x_train = x_train.astype('int32')
    # x_test = x_test.astype('int32')
    # print(f"Data type of x_train: {x_train.dtype}") # Debug print: Check x_train dtype

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
    print(f"Data type of input_sentence: {input_sentence.dtype}") # Debug print: Check input_sentence dtype
    
    # One-hot encode the input
    input_vec = Embedding(16, hidden_dim)(input_sentence)
    h = GCNN(residual=True)(input_vec)
    h = GCNN(residual=True)(h)
    h = GlobalAveragePooling1D()(h)

    z_mean = Dense(latent_dim, name='z_mean')(h)
    z_log_var = Dense(latent_dim, name='z_log_var')(h)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.keras.backend.random_normal(shape=(tf.keras.backend.shape(z_mean)[0], latent_dim), mean=0, stddev=1) # Use tf.keras.backend
        return z_mean + tf.keras.backend.exp(z_log_var / 2) * epsilon # Use tf.keras.backend

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    decoder_hidden = Dense(hidden_dim * n)
    decoder_cnn = GCNN(residual=True)
    decoder_dense = Dense(16, activation='softmax')

    h = decoder_hidden(z)
    h = Reshape((n, hidden_dim))(h)
    h = decoder_cnn(h)
    output = decoder_dense(h)

    class VAE_Loss(Layer): # define a custom layer to calculate the loss
      def __init__(self, **kwargs):
        super(VAE_Loss, self).__init__(**kwargs)

      def call(self, inputs):
        y_true, y_pred, z_mean_loss, z_log_var_loss = inputs # unpack the inputs
        tmp = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        xent_loss = tf.reduce_sum(tmp, axis=-1)
        kl_loss = - 0.5 * tf.reduce_sum(1 + z_log_var_loss - tf.keras.backend.square(z_mean_loss) - tf.keras.backend.exp(z_log_var_loss), axis=-1)
        loss = tf.reduce_mean(xent_loss + kl_loss)
        self.add_loss(loss) # add the loss to the model
        return y_pred # pass the prediction along to the next layer


    output = VAE_Loss()([input_sentence, output, z_mean, z_log_var]) # pass the necessary tensors to the loss layer


    vae = Model(input_sentence, output)
    vae.compile(optimizer='adam') # compile the model, the loss is now handled by the custom layer
    
    vae.summary()

    decoder_input = Input(shape=(latent_dim,))
    _ = decoder_hidden(decoder_input)
    _ = Reshape((n, hidden_dim))(_)
    _ = decoder_cnn(_)
    _output = decoder_dense(_)
    generator = Model(decoder_input, _output)

    def gen():
        r = generator.predict(np.random.randn(1, latent_dim))[0]
        r = r.argmax(axis=1)
        print(r)
        return r

    class Evaluate(Callback):
        def __init__(self):
            self.log = []

        def on_epoch_end(self, epoch, logs=None):
            generated_output_tokens = gen() # Get the generated output (NumPy array)
            self.log.append(generated_output_tokens)

            print(f"Type of generated_output_tokens in Evaluate: {type(generated_output_tokens)}") # Debug print

            gen_address = ""
            count = 0
            gen_address_list = [str(hex(i))[-1] for i in generated_output_tokens] # Use generated_output_tokens
            for i in gen_address_list:
                count += 1
                gen_address += i
                if count % 4 == 0:
                    gen_address += ":"
            gen_address = gen_address[:-1]
            print(gen_address)


    evaluator = Evaluate()

    # Create target data by shifting x_train by one
    x_train_target = np.concatenate((x_train[:,1:], np.zeros((x_train.shape[0],1), dtype='int32')), axis=1)
    vae.fit(x_train,
            x_train_target, # use the target data here
            shuffle=True,
            epochs=3,
            batch_size=64,
            callbacks=[evaluator]
            )

    vae.save_weights('models/gcnn_vae.model.weights.h5')

    for i in range(20):
        r = gen()
        print(f"Type of r in final generation loop: {type(r)}") # Debug print
        gen_address = ""
        count = 0
        gen_address_list = [str(hex(val))[-1] for val in r] # Use r here
        for val in gen_address_list:
            count += 1
            gen_address += val
            if count % 4 == 0:
                gen_address += ":"
        gen_address = gen_address[:-1]
        print(gen_address)


if __name__ == "__main__":
    run_model()
