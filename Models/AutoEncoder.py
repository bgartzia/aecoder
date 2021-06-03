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


    def call(self, inputs):
        """ Returns the difference between input and autoencoder output.
            It is this function which it has to be minimized
        """

        hidden_pipeline = self.encoder(inputs)
        outputs = self.decoder(hidden_pipeline)
        
        return (outputs - inputs)


    def get_latent_vector(self, inputs):
        """ Returns the values stored in the latent space of the autoencoder
        """

        return self.encoder(inputs)

    def get_decoder_out(self, inputs):
        """ Returns the output of the decoder, without substracting the input.
        """

        hidden_pipeline = self.encoder(inputs)

        return self.decoder(hidden_pipeline)

    def get_segmented_anomalies(self, inputs):
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
        
        
            




