import os
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, ReLU,
                                     MaxPool2D, Dense, Flatten,
                                     Conv2DTranspose, UpSampling2D)
from tensorflow import reshape as tf_reshape


class AutoEncoder(Model):




    def __init__(self, input_shape, layers_filter_sizes, latent_space_dim, 
                 segmentation_thresh = None):
        """ Input_shape: 3 dimensional tuple. h,w,channels or som like it
            layers_filter_sizes: list of the filter sizes of the encoder,
                                 and thus of its symmetrical decoder.
            latent_space_dim: dimension of the latent space of the autoencoder
            segmentation_thresh: Number used for extracting detected anomalies.
        """

        super(AutoEncoder, self).__init__()
        self.seg_thresh = segmentation_thresh
        self.in_shape = input_shape
        self.latent_dim = latent_space_dim

        x = Input(shape=input_shape)
        
        # Build encoder layers
        hidden_pipeline = x
        for layer_size in layers_filter_sizes:
            hidden_pipeline = Conv2D(filters=layer_size, kernel_size=3,
                                     strides=(1,1),
                                     padding='same')(hidden_pipeline)
            hidden_pipeline = BatchNormalization()(hidden_pipeline)
            hidden_pipeline = ReLU()(hidden_pipeline)
            hidden_pipeline = MaxPool2D((2,2))(hidden_pipeline)

        encoded = hidden_pipeline

        # Build latent space
        unfold = Flatten()(encoded)
        latent = Dense(latent_space_dim)(unfold)

        # Map latent space to decoder input
        ### Define decoder input
        latent_out = Input(shape=latent.shape[1:])
        ### Use a dense layer to map the latent space dim to the
        ### Symmetric autoencoders decoder input dim
        flatt_decoder_input = Dense(unfold.shape[1])(latent_out)
        # TODO: add activation function here to flatt_decoder_input
        decoder_input_shape = encoded.get_shape().as_list()
        decoder_input_shape[0] = -1
        # Reshape a flattened vector into a convolutionable struct
        decoder_input = tf_reshape(flatt_decoder_input, decoder_input_shape)

        # Decoder is reversed encoder
        layers_filter_sizes.reverse()
        # Build decoder layers
        hidden_pipeline = decoder_input
        for layer_size in layers_filter_sizes:
            hidden_pipeline = UpSampling2D((2,2))(hidden_pipeline)
            hidden_pipeline = Conv2DTranspose(filters=layer_size, kernel_size=3,
                                              strides=(1,1),
                                              padding='same')(hidden_pipeline)
            hidden_pipeline = BatchNormalization()(hidden_pipeline)
            hidden_pipeline = ReLU()(hidden_pipeline)

        decoded = Conv2DTranspose(filters=input_shape[-1],
                                  kernel_size=3,
                                  strides=(1,1),
                                  padding='same')(hidden_pipeline)

        self.encoder = Model(inputs=x, outputs=latent)
        self.decoder = Model(inputs=latent_out, outputs=decoded)

        self.__code_2_call = {'EXTRACT_LSPACES':self.get_latent_vector,
                              'EXTRACT_RAW_OUT':self.get_decoder_out,
                              'EXTRACT_DIFFS':self,
                              'EXTRACT_SEGMENTS':self.get_segmented_anomalies,
                              'EXTRACT_TOT_ERROR':self.get_total_error
                             }


    def __call__(self, inputs, **kwargs):
        """ Returns the difference between input and autoencoder output.
            It is this function which it has to be minimized
        """

        hidden_pipeline = self.encoder(inputs)
        outputs = self.decoder(hidden_pipeline)
        
        return (outputs - inputs)


    def get_latent_vector(self, inputs, **kwargs):
        """ Returns the values stored in the latent space of the autoencoder
        """

        return self.encoder(inputs)


    def get_decoder_out(self, inputs, **kwargs):
        """ Returns the output of the decoder, without substracting the input.
        """

        hidden_pipeline = self.encoder(inputs)

        return self.decoder(hidden_pipeline)


    def get_segmented_anomalies(self, inputs, **kwargs):
        """ Returns the segmented hot-spots, and their intensity
        """

        # Extract difference between the input and the autoencoder output
        ae_diff = self(inputs)
        
        if self.seg_thresh is not None:
            # Extract segmented mask
            mask = tf.cast(ae_diff > self.seg_thresh, dtype=tf.float32)
        else:
            # If threshold hasn't been defined, just return it all
            mask = tf.constant(1, shape=ae_diff.shape)
        # Return product between values and mask. Thus, not only the segmented
        # ROIs are returned but also the intensity of the detection, or the 
        # "Confidence", or the temperature of the _HOT-SPOTS_
        return tf.multiply(ae_diff, mask)


    def get_total_error(self, inputs, **kwargs):
        """ Calculates the total error between the input and the input for each
            of the images
        """

        diffs = self(inputs)
        #TODO@BGARCIA: should the loss function be parametrizable??
        errors = tf.map_fn(tf.nn.l2_loss, diffs)
        return errors


    def save(self, directory, name_root):
        """ Saves encoder and decoder models in the specified directory.
            directory: place where the models are stored.
            name_root: name for the files. They will be followed by encoder or
                       decoder.
        """

        if not os.path.exists(directory):
            os.makedirs(directory)

        enc_path = os.path.join(directory, f'{name_root}_encoder.h5')
        dec_path = os.path.join(directory, f'{name_root}_decoder.h5')

        self.encoder.save(enc_path)
        self.decoder.save(dec_path)


    def load(self, directory, name_root):
        """ Loads encoder and decoder models from the specified directory.
            directory: place where the models are stored.
            name_root: name for the files. They will be followed by encoder or
                       decoder.
        """

        enc_path = os.path.join(directory, f'{name_root}_encoder.h5')
        dec_path = os.path.join(directory, f'{name_root}_decoder.h5')

        self.encoder.load_weights(enc_path)
        self.decoder.load_weights(dec_path)


    def code_call(self, data, code):
        """ Calls an specific extraction method depending on the extraction code.
        """

        return self.__code_2_call[code](data)


