
#%% [markdown]
### 1. Import Libraries
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.layers import (
    Conv1D,
    Flatten,
    Dense,
    Conv1DTranspose,
    Reshape,
    Input,
    Layer,
)

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from scipy.linalg import sqrtm
from tensorflow import keras


# Feature Extraction for FID Calculation
class FIDCalculator:
    def __init__(self, model):
        self.model = model
        self.real_features = []
        self.generated_features = []
        
    def update(self, real_data, generated_data):
        # Extract features using TensorFlow operations
        _, _, real_latent = self.model.encoder(real_data)
        _, _, gen_latent = self.model.encoder(generated_data)
        
        self.real_features.append(real_latent)
        self.generated_features.append(gen_latent)
    
    def compute_fid(self):
        # Convert to numpy only when calculating
        real = tf.concat(self.real_features, axis=0).numpy()
        gen = tf.concat(self.generated_features, axis=0).numpy()
        
        if len(real) < 2 or len(gen) < 2:
            return float('inf')  # Handle small batch sizes
        
        mu_real, sigma_real = np.mean(real, axis=0), np.cov(real, rowvar=False)
        mu_gen, sigma_gen = np.mean(gen, axis=0), np.cov(gen, rowvar=False)

        ssdiff = np.sum((mu_real - mu_gen)**2)
        covmean = sqrtm(sigma_real.dot(sigma_gen))
        covmean = covmean.real if np.iscomplexobj(covmean) else covmean
            
        fid = ssdiff + np.trace(sigma_real + sigma_gen - 2*covmean)
        return fid
    

def temporal_correlation_score(real, generated):
    # TensorFlow-based correlation calculation
    def _tf_correlation(data):
        data = tf.transpose(data, perm=[0, 2, 1])  # Convert to [batch, features, time]
        mean = tf.reduce_mean(data, axis=-1, keepdims=True)
        std = tf.math.reduce_std(data, axis=-1, keepdims=True)
        norm_data = (data - mean) / (std + 1e-8)
        corr = tf.matmul(norm_data, norm_data, transpose_b=True) / tf.cast(tf.shape(data)[-1], tf.float32)
        return corr
    
    real_corr = _tf_correlation(real)
    gen_corr = _tf_correlation(generated)
    return tf.reduce_mean(tf.abs(real_corr - gen_corr))



# Add the custom Sampling layer definition
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class TrendLayer(Layer):
    def __init__(self, feat_dim, trend_poly, seq_len, **kwargs):
        super(TrendLayer, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.trend_poly = trend_poly
        self.seq_len = seq_len
        self.trend_dense1 = Dense(
            self.feat_dim * self.trend_poly, activation="relu", name="trend_params"
        )
        self.trend_dense2 = Dense(self.feat_dim * self.trend_poly, name="trend_params2")
        self.reshape_layer = Reshape(target_shape=(self.feat_dim, self.trend_poly))

    def call(self, z):
        trend_params = self.trend_dense1(z)
        trend_params = self.trend_dense2(trend_params)
        trend_params = self.reshape_layer(trend_params)  # shape: N x D x P

        lin_space = (
            tf.range(0, float(self.seq_len), 1) / self.seq_len
        )  # shape of lin_space: 1d tensor of length T
        poly_space = tf.stack(
            [lin_space ** float(p + 1) for p in range(self.trend_poly)], axis=0
        )  # shape: P x T

        trend_vals = tf.matmul(trend_params, poly_space)  # shape (N, D, T)
        trend_vals = tf.transpose(trend_vals, perm=[0, 2, 1])  # shape: (N, T, D)
        trend_vals = tf.cast(trend_vals, tf.float32)

        return trend_vals

class SeasonalLayer(Layer):
    def __init__(self, feat_dim, seq_len, custom_seas, **kwargs):
        super(SeasonalLayer, self).__init__(**kwargs)
        self.feat_dim = feat_dim
        self.seq_len = seq_len
        self.custom_seas = custom_seas
        self.dense_layers = [
            Dense(feat_dim * num_seasons, name=f"season_params_{i}")
            for i, (num_seasons, len_per_season) in enumerate(custom_seas)
        ]
        self.reshape_layers = [
            Reshape(target_shape=(feat_dim, num_seasons))
            for num_seasons, len_per_season in custom_seas
        ]

    def _get_season_indexes_over_seq(self, num_seasons, len_per_season):
        season_indexes = tf.range(num_seasons)[:, None] + tf.zeros(
            (num_seasons, len_per_season), dtype=tf.int32
        )
        season_indexes = tf.reshape(season_indexes, [-1])
        # Ensure the length matches seq_len
        season_indexes = tf.tile(season_indexes, [self.seq_len // len_per_season + 1])[
            : self.seq_len
        ]
        return season_indexes

    def call(self, z):
        N = tf.shape(z)[0]
        ones_tensor = tf.ones(shape=[N, self.feat_dim, self.seq_len], dtype=tf.int32)

        all_seas_vals = []
        for i, (num_seasons, len_per_season) in enumerate(self.custom_seas):
            season_params = self.dense_layers[i](z)  # shape: (N, D * S)
            season_params = self.reshape_layers[i](season_params)  # shape: (N, D, S)

            season_indexes_over_time = self._get_season_indexes_over_seq(
                num_seasons, len_per_season
            )  # shape: (T, )

            dim2_idxes = ones_tensor * tf.reshape(
                season_indexes_over_time, shape=(1, 1, -1)
            )  # shape: (N, D, T)
            season_vals = tf.gather(
                season_params, dim2_idxes, batch_dims=-1
            )  # shape (N, D, T)

            all_seas_vals.append(season_vals)

        all_seas_vals = K.stack(all_seas_vals, axis=-1)  # shape: (N, D, T, S)
        all_seas_vals = tf.reduce_sum(all_seas_vals, axis=-1)  # shape (N, D, T)
        all_seas_vals = tf.transpose(all_seas_vals, perm=[0, 2, 1])  # shape (N, T, D)
        return all_seas_vals

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.seq_len, self.feat_dim)


# Basic VAE model
@keras.saving.register_keras_serializable(package="MyCustomModels")
class VAE_CNN(Model):
    def __init__(self, seq_len, feat_dim, hidden_layer_sizes=[50, 100, 200], latent_dim=8, reconstruction_wt=3.0, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.reconstruction_wt = reconstruction_wt
        self.hidden_layer_sizes = hidden_layer_sizes
        
        # Build encoder
        self.encoder = self._get_encoder()
        
        # Build decoder
        self.decoder = self._get_decoder()
        self.compile(optimizer=Adam())

    def _get_encoder(self):
        encoder_inputs = layers.Input(
            shape=(self.seq_len, self.feat_dim), name="encoder_input"
        )
        x = encoder_inputs
        for i, num_filters in enumerate(self.hidden_layer_sizes):
            x = Conv1D(
                filters=num_filters,
                kernel_size=3,
                strides=2,
                activation="relu",
                padding="same",
                name=f"enc_conv_{i}",
            )(x)

        x = Flatten(name="enc_flatten")(x)

        # save the dimensionality of this last dense layer before the hidden state layer. We need it in the decoder.
        self.encoder_last_dense_dim = x.shape[-1]

        z_mean = Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(x)

        encoder_output = Sampling()([z_mean, z_log_var])
        self.encoder_output = encoder_output

        encoder = Model(
            encoder_inputs, [z_mean, z_log_var, encoder_output], name="encoder"
        )
        # encoder.summary()
        return encoder

    def _get_decoder(self):
        decoder_inputs = layers.Input(shape=(self.latent_dim,), name="decoder_input")

        x = decoder_inputs
        x = Dense(self.encoder_last_dense_dim, name="dec_dense", activation="relu")(x)
        x = Reshape(target_shape=(-1, self.hidden_layer_sizes[-1]), name="dec_reshape")(
            x
        )

        for i, num_filters in enumerate(reversed(self.hidden_layer_sizes[:-1])):
            x = Conv1DTranspose(
                filters=num_filters,
                kernel_size=3,
                strides=2,
                padding="same",
                activation="relu",
                name=f"dec_deconv_{i}",
            )(x)

        # last de-convolution
        x = Conv1DTranspose(
            filters=self.feat_dim,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu",
            name=f"dec_deconv__{i+1}",
        )(x)

        x = Flatten(name="dec_flatten")(x)
        x = Dense(self.seq_len * self.feat_dim, name="decoder_dense_final", activation='sigmoid')(x)
        self.decoder_outputs = Reshape(target_shape=(self.seq_len, self.feat_dim))(x)
        decoder = Model(decoder_inputs, self.decoder_outputs, name="decoder")
        return decoder
    
    def call(self, X):
        z_mean, _, _ = self.encoder(X)
        x_decoded = self.decoder(z_mean)
        if len(x_decoded.shape) == 1:
            x_decoded = x_decoded.reshape((1, -1))
        return x_decoded
    
    def _get_reconstruction_loss(self, X, X_recons):
        def get_reconst_loss_by_axis(X, X_c, axis):
            x_r = tf.reduce_mean(X, axis=axis)
            x_c_r = tf.reduce_mean(X_recons, axis=axis)
            err = tf.math.squared_difference(x_r, x_c_r)
            loss = tf.reduce_sum(err)
            return loss

        # overall
        err = tf.math.squared_difference(X, X_recons)
        reconst_loss = tf.reduce_sum(err)

        reconst_loss += get_reconst_loss_by_axis(X, X_recons, axis=[2])  # by time axis
        # reconst_loss += get_reconst_loss_by_axis(X, X_recons, axis=[1])    # by feature axis
        return reconst_loss
    
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            # Forward pass        
            z_mean, z_log_var, z = self.encoder(data)

            reconstruction = self.decoder(z)

            reconstruction_loss = self._get_reconstruction_loss(data, reconstruction)
            
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_sum(tf.reduce_sum(kl_loss, axis=1))
            # kl_loss = kl_loss / self.latent_dim

            total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss
            
        
        # Backpropagation
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        return {
            "total_loss": total_loss,
            "recon_loss": reconstruction_loss,
            "kl_loss": kl_loss
        }


# TimeVAE model
class TIME_VAE(Model):
    def __init__(self, seq_len, feat_dim, hidden_layer_sizes=[50, 100, 200], latent_dim=8, 
                 reconstruction_wt=3.0, trend_poly=0, custom_seas=None, use_residual_conn=True, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.reconstruction_wt = reconstruction_wt
        self.hidden_layer_sizes = hidden_layer_sizes
        self.trend_poly = trend_poly
        self.custom_seas = custom_seas
        self.use_residual_conn = use_residual_conn
        
        # Build encoder
        self.encoder = self._get_encoder()
        
        # Build decoder
        self.decoder = self._get_decoder()
        self.compile(optimizer=Adam())

    def _get_encoder(self):
        encoder_inputs = layers.Input(
            shape=(self.seq_len, self.feat_dim), name="encoder_input"
        )
        x = encoder_inputs
        for i, num_filters in enumerate(self.hidden_layer_sizes):
            x = Conv1D(
                filters=num_filters,
                kernel_size=3,
                strides=2,
                activation="relu",
                padding="same",
                name=f"enc_conv_{i}",
            )(x)

        x = Flatten(name="enc_flatten")(x)

        # save the dimensionality of this last dense layer before the hidden state layer. We need it in the decoder.
        self.encoder_last_dense_dim = x.shape[-1]

        z_mean = Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(x)

        encoder_output = Sampling()([z_mean, z_log_var])
        self.encoder_output = encoder_output

        encoder = Model(
            encoder_inputs, [z_mean, z_log_var, encoder_output], name="encoder"
        )
        # encoder.summary()
        return encoder

    def _get_decoder(self):
        decoder_inputs = layers.Input(shape=(self.latent_dim,), name="decoder_input")

        outputs = None
        outputs = self.level_model(decoder_inputs)
        # trend polynomials
        if self.trend_poly is not None and self.trend_poly > 0:
            trend_vals = TrendLayer(self.feat_dim, self.trend_poly, self.seq_len)(
                decoder_inputs
            )
            outputs = trend_vals if outputs is None else outputs + trend_vals

        # custom seasons
        if self.custom_seas is not None and len(self.custom_seas) > 0:
            # cust_seas_vals = self.custom_seasonal_model(decoder_inputs)
            cust_seas_vals = SeasonalLayer(
                feat_dim=self.feat_dim,
                seq_len=self.seq_len,
                custom_seas=self.custom_seas,
            )(decoder_inputs)
            outputs = cust_seas_vals if outputs is None else outputs + cust_seas_vals

        if self.use_residual_conn:
            residuals = self._get_decoder_residual(decoder_inputs)
            outputs = residuals if outputs is None else outputs + residuals

        if outputs is None:
            raise ValueError(
                "Error: No decoder model to use. "
                "You must use one or more of:"
                "trend, generic seasonality(ies), custom seasonality(ies), "
                "and/or residual connection. "
            )

        decoder = Model(decoder_inputs, [outputs], name="decoder")
        return decoder

    def level_model(self, z):
        level_params = Dense(self.feat_dim, name="level_params", activation="relu")(z)
        level_params = Dense(self.feat_dim, name="level_params2")(level_params)
        level_params = Reshape(target_shape=(1, self.feat_dim))(
            level_params
        )  # shape: (N, 1, D)

        ones_tensor = tf.ones(
            shape=[1, self.seq_len, 1], dtype=tf.float32
        )  # shape: (1, T, D)

        level_vals = level_params * ones_tensor
        # print('level_vals', tf.shape(level_vals))
        return level_vals
    
    def _get_decoder_residual(self, x):
        x = Dense(self.encoder_last_dense_dim, name="dec_dense", activation="relu")(x)
        x = Reshape(target_shape=(-1, self.hidden_layer_sizes[-1]), name="dec_reshape")(
            x
        )

        for i, num_filters in enumerate(reversed(self.hidden_layer_sizes[:-1])):
            x = Conv1DTranspose(
                filters=num_filters,
                kernel_size=3,
                strides=2,
                padding="same",
                activation="relu",
                name=f"dec_deconv_{i}",
            )(x)

        # last de-convolution
        x = Conv1DTranspose(
            filters=self.feat_dim,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu",
            name=f"dec_deconv__{i+1}",
        )(x)

        x = Flatten(name="dec_flatten")(x)
        x = Dense(self.seq_len * self.feat_dim, name="decoder_dense_final", activation='sigmoid')(x)
        residuals = Reshape(target_shape=(self.seq_len, self.feat_dim))(x)
        return residuals

    def call(self, X):
        z_mean, _, _ = self.encoder(X)
        x_decoded = self.decoder(z_mean)
        if len(x_decoded.shape) == 1:
            x_decoded = x_decoded.reshape((1, -1))
        return x_decoded
    
    def _get_reconstruction_loss(self, X, X_recons):
        def get_reconst_loss_by_axis(X, X_c, axis):
            x_r = tf.reduce_mean(X, axis=axis)
            x_c_r = tf.reduce_mean(X_recons, axis=axis)
            err = tf.math.squared_difference(x_r, x_c_r)
            loss = tf.reduce_sum(err)
            return loss

        # overall
        err = tf.math.squared_difference(X, X_recons)
        reconst_loss = tf.reduce_sum(err)

        reconst_loss += get_reconst_loss_by_axis(X, X_recons, axis=[2])  # by time axis
        # reconst_loss += get_reconst_loss_by_axis(X, X_recons, axis=[1])    # by feature axis
        return reconst_loss
    
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            # Forward pass        
            z_mean, z_log_var, z = self.encoder(data)

            reconstruction = self.decoder(z)

            reconstruction_loss = self._get_reconstruction_loss(data, reconstruction)
            
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_sum(tf.reduce_sum(kl_loss, axis=1))
            # kl_loss = kl_loss / self.latent_dim

            total_loss = self.reconstruction_wt * reconstruction_loss + kl_loss
            
        
        # Backpropagation
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        return {
            "total_loss": total_loss,
            "recon_loss": reconstruction_loss,
            "kl_loss": kl_loss
        }


#%% [markdown]
### Fixed Metric Monitoring with Eager Execution
#%%
class VAEMonitor(tf.keras.callbacks.Callback):
    def __init__(self, val_data, freq=5):
        super().__init__()
        self.val_data = val_data
        self.freq = freq
        self.fid_history = []
        self.corr_history = []

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.freq == 0:
            # Calculate FID on validation set
            real_samples = self.val_data[:100]  # Use subset for efficiency
            generated_samples = self.model.predict(real_samples)
            
            # Get latent representations
            z_mean_real, _, _ = self.model.encoder.predict(real_samples, verbose=0)
            z_mean_gen, _, _ = self.model.encoder.predict(generated_samples, verbose=0)
            
            # Calculate FID
            fid = self.calculate_fid(z_mean_real, z_mean_gen)
            self.fid_history.append(fid)
            
            # Calculate temporal correlation
            corr_score = self.temporal_correlation(real_samples, generated_samples)
            self.corr_history.append(corr_score)
            
            print(f"\nEpoch {epoch+1} Metrics:")
            print(f"FID: {fid:.2f}")
            print(f"Temporal Correlation Score: {corr_score:.4f}")

    def calculate_fid(self, real, generated):
        # Convert to numpy arrays
        real = real.numpy() if tf.is_tensor(real) else real
        generated = generated.numpy() if tf.is_tensor(generated) else generated
        
        mu_real, sigma_real = np.mean(real, axis=0), np.cov(real, rowvar=False)
        mu_gen, sigma_gen = np.mean(generated, axis=0), np.cov(generated, rowvar=False)
        
        ssdiff = np.sum((mu_real - mu_gen)**2)
        covmean = sqrtm(sigma_real @ sigma_gen).real
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = ssdiff + np.trace(sigma_real + sigma_gen - 2*covmean)
        return float(fid)

    def temporal_correlation(self, real, generated):
        # Use TensorFlow operations for compatibility
        real = tf.convert_to_tensor(real)
        generated = tf.convert_to_tensor(generated)
        
        def _calc_corr(data):
            data = tf.transpose(data, [0, 2, 1])  # [batch, features, time]
            corr = tf.matmul(data, data, transpose_b=True) / tf.cast(tf.shape(data)[2], tf.float32)
            return corr
        
        real_corr = _calc_corr(real)
        gen_corr = _calc_corr(generated)
        return tf.reduce_mean(tf.abs(real_corr - gen_corr)).numpy()


