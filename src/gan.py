
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import wasserstein_distance


def build_cwgan_generator(latent_dim, window_size, num_features, num_conditions):
    noise = layers.Input(shape=(latent_dim,))
    condition = layers.Input(shape=(window_size, num_conditions))  # Input condition
    
    # Process condition with Conv1D
    c = layers.Conv1D(32, 64, padding='same')(condition)
    c = layers.LeakyReLU(0.2)(c)
    c = layers.Conv1D(64, 12, padding='same')(c)
    c = layers.LeakyReLU(0.2)(c)
    c = layers.GlobalAveragePooling1D()(c)  # Better than flattening for temporal features
    
    # Concatenate noise and processed condition
    combined = layers.concatenate([noise, c])
    
    # Original generator layers
    x = layers.Dense((window_size//8) * 256, activation='relu')(combined)
    x = layers.Reshape(((window_size//8), 256))(x)
    
    x = layers.Conv1DTranspose(128, 8, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv1DTranspose(64, 8, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    x = layers.Conv1DTranspose(32, 8, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    outputs = layers.Conv1D(num_features, 8, padding='same', activation='sigmoid')(x)
    
    return keras.Model([noise, condition], outputs, name="generator")


def build_cwgan_discriminator(window_size, num_features, num_conditions):
    data_input = layers.Input(shape=(window_size, num_features))
    condition_input = layers.Input(shape=(window_size, num_conditions))
    
    # Concatenate data and condition
    combined = layers.concatenate([data_input, condition_input], axis=-1)
    
    x = layers.Conv1D(64, 8, strides=2, padding='same')(combined)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv1D(128, 8, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv1D(256, 8, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Flatten()(x)
    
    outputs = layers.Dense(1)(x)
    
    return keras.Model([data_input, condition_input], outputs, name="discriminator")




def build_wgan_generator(latent_dim, window_size, num_features):
    noise = layers.Input(shape=(latent_dim,))
    x = layers.Dense((window_size//8) * 256, activation='relu')(noise)
    # the 128 noise vector is projected into a higher-dimensional space (768 units) to create an initial structure (3 timesteps, 256 channels) that can be upsampled.
    x = layers.Reshape(((window_size//8), 256))(x)
    
    # Upsampling layers
    x = layers.Conv1DTranspose(256, 8, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv1DTranspose(128, 8, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv1DTranspose(64, 8, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)    
        
    outputs = layers.Conv1D(num_features, 8, padding='same', activation='sigmoid')(x)
    
    model = keras.Model(inputs=noise, outputs=outputs, name="generator")
    return model


def build_wgan_discriminator(window_size, num_features):
    inputs = layers.Input(shape=(window_size, num_features))
    x = layers.Conv1D(64, 8, strides=2, padding='same')(inputs)
    x = layers.LeakyReLU(0.2)(x)
    #x = layers.MaxPooling1D(1)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv1D(128, 8, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    #x = layers.MaxPooling1D(1)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv1D(256, 8, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(1)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="discriminator")
    return model


# CWGAN-GP Model
class CWGAN_GP(keras.Model):
    def __init__(self, critic, generator, latent_dim,  window_size, num_conditions, n_critic=5, gp_weight=10.0):
        super().__init__()
        self.critic = critic
        self.generator = generator
        self.latent_dim = latent_dim
        self.window_size = window_size
        self.num_conditions = num_conditions
        self.n_critic = n_critic
        self.gp_weight = gp_weight
               
        # Build model with explicit input shapes
        self.build([(None, latent_dim), (None, window_size, num_conditions)])

    def build(self, input_shapes):
        # Initialize with proper input shapes
        noise_shape, condition_shape = input_shapes
        self.noise_input = keras.Input(shape=noise_shape[1:], name="noise_input")
        self.condition_input = keras.Input(shape=condition_shape[1:], name="condition_input")
        super().build(input_shapes)
    
    def call(self, inputs, training=None):
        # Unpack noise and conditions
        noise, conditions = inputs
        return self.generator([noise, conditions], training=training)
    
    def get_config(self):
        return {
            "latent_dim": self.latent_dim,
            "window_size": self.window_size,
            "num_conditions": self.num_conditions,
            "n_critic": self.n_critic,
            "gp_weight": self.gp_weight,
            "critic_config": self.critic.get_config(),
            "generator_config": self.generator.get_config()
        }

    @classmethod
    def from_config(cls, config):
        critic = keras.models.Model.from_config(config["critic_config"])
        generator = keras.models.Model.from_config(config["generator_config"])
        return cls(
            critic=critic,
            generator=generator,
            latent_dim=config["latent_dim"],
            window_size=config["window_size"],
            num_conditions=config["num_conditions"],
            n_critic=config["n_critic"],
            gp_weight=config["gp_weight"]
        )


    def compile(self, c_optimizer, g_optimizer):
        super().compile()
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer
        self.c_loss_metric = keras.metrics.Mean(name="c_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    def gradient_penalty(self, real_data, fake_data, real_conditions):
        batch_size = tf.shape(real_data)[0]
        alpha = tf.random.uniform([batch_size, 1, 1], 0.0, 1.0)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.critic([interpolated, real_conditions], training=True)
        
        gradients = tape.gradient(pred, interpolated)
        gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2]))
        gp = tf.reduce_mean((gradients_norm - 1.0) ** 2)
        return gp
    
    def train_step(self, data):
        real_data, real_conditions = data  # Unpack data and conditions

        # Train critic
        for _ in range(self.n_critic):
            noise = tf.random.normal([tf.shape(real_data)[0], self.latent_dim])
            with tf.GradientTape() as tape:
                fake_data = self.generator([noise, real_conditions], training=True)
                real_pred = self.critic([real_data, real_conditions], training=True)
                fake_pred = self.critic([fake_data, real_conditions], training=True)
                
                c_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred)
                gp = self.gradient_penalty(real_data, fake_data, real_conditions)
                c_total_loss = c_loss + self.gp_weight * gp
            
            c_grads = tape.gradient(c_total_loss, self.critic.trainable_weights)
            self.c_optimizer.apply_gradients(zip(c_grads, self.critic.trainable_weights))
            self.c_loss_metric.update_state(c_total_loss)

        # Train generator
        noise = tf.random.normal([tf.shape(real_data)[0], self.latent_dim])
        with tf.GradientTape() as tape:
            fake_data = self.generator([noise, real_conditions], training=True)
            fake_pred = self.critic([fake_data, real_conditions], training=True)
            g_loss = -tf.reduce_mean(fake_pred)
        
        g_grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))
        self.g_loss_metric.update_state(g_loss)

        return {m.name: m.result() for m in [self.c_loss_metric, self.g_loss_metric]}
   

# WGAN-GP Model
class WGAN_GP(keras.Model):
    def __init__(self, critic, generator, latent_dim, n_critic=5, gp_weight=10.0):
        super().__init__()
        self.critic = critic
        self.generator = generator
        self.latent_dim = latent_dim
        self.n_critic = n_critic
        self.gp_weight = gp_weight
              
        # Explicit input specification
        self.build((None, latent_dim))

    def build(self, input_shape):
        # Ensure generator is built with concrete input shape
        if not self.generator.built:
            self.generator.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        return self.generator(inputs, training=training)

    def get_config(self):
        return {
            "critic_config": self.critic.get_config(),
            "generator_config": self.generator.get_config(),
            "latent_dim": self.latent_dim,
            "n_critic": self.n_critic,
            "gp_weight": self.gp_weight
        }
    
    @classmethod
    def from_config(cls, config):
        # Rebuild submodels from config
        critic = keras.models.Model.from_config(config["critic_config"])
        generator = keras.models.Model.from_config(config["generator_config"])
        return cls(
            critic=critic,
            generator=generator,
            latent_dim=config["latent_dim"],
            n_critic=config["n_critic"],
            gp_weight=config["gp_weight"]
        )

    # Add compile serialization (important for loading)
    def get_compile_config(self):
        return {
            "c_optimizer": keras.optimizers.serialize(self.c_optimizer),
            "g_optimizer": keras.optimizers.serialize(self.g_optimizer)
        }
    
    def compile_from_config(self, config):
        self.compile(
            c_optimizer=keras.optimizers.deserialize(config["c_optimizer"]),
            g_optimizer=keras.optimizers.deserialize(config["g_optimizer"])
        )


    def compile(self, c_optimizer, g_optimizer):
        super().compile()
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer
        self.c_loss_metric = keras.metrics.Mean(name="c_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
    
    def gradient_penalty(self, real, fake):
        batch_size = tf.shape(real)[0]
        alpha = tf.random.uniform([batch_size, 1, 1], 0., 1.)
        interpolated = alpha * real + (1 - alpha) * fake
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = self.critic(interpolated)
        
        gradients = tape.gradient(pred, interpolated)
        
        # Feature-specific gradient penalty
        #gradients_per_feature = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
        #gp = tf.reduce_mean((gradients_per_feature - 1.0) ** 2)
        
        gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2]))
        gp = tf.reduce_mean((gradients_norm - 1.0) ** 2)
        
        return gp
   
    def train_step(self, real_data):
        
        for _ in range(self.n_critic):  # Train Critic More Often
            noise = tf.random.normal([tf.shape(real_data)[0], self.latent_dim])
            with tf.GradientTape() as tape:
                fake_data = self.generator(noise, training=True)
                real_pred = self.critic(real_data, training=True)
                fake_pred = self.critic(fake_data, training=True)

                c_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred)
                gp = self.gradient_penalty(real_data, fake_data)
                c_total_loss = c_loss + self.gp_weight * gp

            c_grads = tape.gradient(c_total_loss, self.critic.trainable_weights)
            self.c_optimizer.apply_gradients(zip(c_grads, self.critic.trainable_weights))
            self.c_loss_metric.update_state(c_total_loss)
            
    
        noise = tf.random.normal([tf.shape(real_data)[0], self.latent_dim])
        with tf.GradientTape() as tape:
            fake_data = self.generator(noise, training=True)
            fake_pred = self.critic(fake_data, training=True)
            g_loss = -tf.reduce_mean(fake_pred)
            
        g_grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))
        self.g_loss_metric.update_state(g_loss)

        return {m.name: m.result() for m in [self.c_loss_metric, self.g_loss_metric]}



class CWGANMonitor(keras.callbacks.Callback):
    def __init__(self, real_data, real_conditions, latent_dim, window_size, num_features, every_n=10):
        super().__init__()
        self.real_data = real_data
        self.real_conditions = real_conditions
        self.latent_dim = latent_dim
        self.window_size = window_size
        self.num_features = num_features
        self.every_n = every_n
        self.best_fid = float('inf')
        self.fid_history = []
        self.temp_corr_history = []
        self.real_corr_history = []
        self.wasserstein_dist_history = []
        
        # Build feature extractor for fid calculation
        self.feature_extractor = self.build_time_series_feature_extractor()
        
    def build_time_series_feature_extractor(self):
        """CNN-based feature extractor for multivariate time-series data."""
        inputs = keras.Input(shape=(self.window_size, self.num_features))
        
        x = layers.Conv1D(64, kernel_size=3, strides=1, activation='relu', padding='same')(inputs)
        x = layers.Conv1D(128, kernel_size=3, strides=1, activation='relu', padding='same')(x)  # Second Conv1D layer
        x = layers.GlobalAveragePooling1D()(x)  # Pooling to get a fixed-size feature vector
        
        model = keras.Model(inputs, x, name="TimeSeriesFeatureExtractor")
        return model
    
    def calculate_fid(self, real, synthetic):
        """Frechet Inception Distance using critic's features"""
        real_features = self.feature_extractor.predict(real, verbose=0)
        syn_features = self.feature_extractor.predict(synthetic, verbose=0)
        
        mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu_syn, sigma_syn = np.mean(syn_features, axis=0), np.cov(syn_features, rowvar=False)
        
        #diff = mu_real - mu_syn
        # Compute squared mean difference
        ssdiff = np.sum((mu_real - mu_syn)**2)
        
        # Compute sqrt of covariance product (handling singular matrix case)
        covmean = linalg.sqrtm(sigma_real @ sigma_syn, disp=False)[0].real
        #if np.iscomplexobj(covmean):
        #    covmean = covmean.real  # Ensure real-valued result
        
        fid = ssdiff + np.trace(sigma_real + sigma_syn - 2*covmean)
        return float(fid)
        #return diff.dot(diff) + np.trace(sigma_real + sigma_syn - 2*covmean)
    
    @staticmethod
    def temporal_correlation_score(data, lag=1):
        """Normalized autocorrelation calculation"""
        if tf.is_tensor(data):
            data = data.numpy()
            
        scores = []
        for sample in data:  # (timesteps, features)
            for feature in range(data.shape[2]):
                series = sample[:, feature]
                if np.var(series) < 1e-6:  # Skip constant series
                    continue
                acf = np.correlate(series - np.mean(series), 
                                 series - np.mean(series), 
                                 mode='full')
                norm_acf = acf[len(series)-1 + lag] / (len(series) * np.var(series))
                scores.append(norm_acf)
        return np.nanmean(scores) if scores else 0.0
    
    def compute_wasserstein_distance(self, real_data: np.ndarray, synthetic_data: np.ndarray):
        """Compute Wasserstein Distance between real and synthetic distributions."""
        if tf.is_tensor(real_data):
            real_data = real_data.numpy()
        if tf.is_tensor(synthetic_data):
            synthetic_data = synthetic_data.numpy() 
            
        # Reshape data to 2D: (samples * time steps, features)
        real_flat = real_data.reshape(-1, real_data.shape[2])  # Shape: (samples * time steps, features)
        synthetic_flat = synthetic_data.reshape(-1, synthetic_data.shape[2])  # Shape: (samples * time steps, features)

        # Compute Wasserstein distance per feature
        wd_per_feature = [wasserstein_distance(real_flat[:, i], synthetic_flat[:, i]) for i in range(real_data.shape[2])]

        # Compute mean Wasserstein distance across all features
        mean_wd = np.mean(wd_per_feature)

        return mean_wd
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n == 0:
            # Get matching number of conditions
            num_samples = min(self.real_data.shape[0], self.real_conditions.shape[0])
            conditions = self.real_conditions[:num_samples]


            # Generate synthetic data with conditions
            noise = tf.random.normal([num_samples, self.latent_dim])
            synthetic = self.model.generator([noise, conditions], training=False) 
            
            # Calculate metrics
            #real_features = self.feature_extractor.predict(self.real_data)
            #fake_features = self.feature_extractor.predict(synthetic)

            #fid = self.calculate_fid(real_features, fake_features)
            fid = self.calculate_fid(self.real_data[:num_samples], synthetic)
            
            real_corr = self.temporal_correlation_score(self.real_data[:num_samples])
            temp_corr = self.temporal_correlation_score(synthetic)
            
            wasserstein_dist = self.compute_wasserstein_distance(self.real_data[:num_samples], synthetic)
            
            # Save best model
            if fid < self.best_fid:
                self.best_fid = fid
                self.model.generator.save_weights("best_generator_weights.h5")
            
            # Store and print
            self.fid_history.append(fid)
            self.temp_corr_history.append(temp_corr)
            self.real_corr_history.append(real_corr)
            self.wasserstein_dist_history.append(wasserstein_dist)
            # Print metrics
            print(f"\nEpoch {epoch+1} Metrics:")
            print(f"FID: {fid:.2f} (Lower better)")
            print(f"TempCorr: {temp_corr:.2f} (Real: {real_corr:.2f})")
            print(f"wasserstein_dist: {wasserstein_dist:.2f} (Lower better)")



class WGANMonitor(keras.callbacks.Callback):
    def __init__(self, real_data, latent_dim, window_size, num_features, every_n=10):
        super().__init__()
        self.real_data = real_data
        self.latent_dim = latent_dim
        self.window_size = window_size
        self.num_features = num_features
        self.every_n = every_n
        self.best_fid = float('inf')
        self.fid_history = []
        self.temp_corr_history = []
        self.real_corr_history = []
        self.wasserstein_dist_history = []
        
        # Build feature extractor for fid calculation
        self.feature_extractor = self.build_time_series_feature_extractor()
        
    def build_time_series_feature_extractor(self):
        """CNN-based feature extractor for multivariate time-series data."""
        inputs = keras.Input(shape=(self.window_size, self.num_features))
        
        x = layers.Conv1D(64, kernel_size=3, strides=1, activation='relu', padding='same')(inputs)
        x = layers.Conv1D(128, kernel_size=3, strides=1, activation='relu', padding='same')(x)  # Second Conv1D layer
        x = layers.GlobalAveragePooling1D()(x)  # Pooling to get a fixed-size feature vector
        
        model = keras.Model(inputs, x, name="TimeSeriesFeatureExtractor")
        return model
    
    def calculate_fid(self, real, synthetic):
        """Frechet Inception Distance using critic's features"""
        real_features = self.feature_extractor.predict(real, verbose=0)
        syn_features = self.feature_extractor.predict(synthetic, verbose=0)
        
        mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        mu_syn, sigma_syn = np.mean(syn_features, axis=0), np.cov(syn_features, rowvar=False)
        
        #diff = mu_real - mu_syn
        # Compute squared mean difference
        ssdiff = np.sum((mu_real - mu_syn)**2)
        
        # Compute sqrt of covariance product (handling singular matrix case)
        covmean = linalg.sqrtm(sigma_real @ sigma_syn, disp=False)[0].real
        #if np.iscomplexobj(covmean):
        #    covmean = covmean.real  # Ensure real-valued result
        
        fid = ssdiff + np.trace(sigma_real + sigma_syn - 2*covmean)
        return float(fid)
        #return diff.dot(diff) + np.trace(sigma_real + sigma_syn - 2*covmean)
    
    @staticmethod
    def temporal_correlation_score(data, lag=1):
        """Normalized autocorrelation calculation"""
        if tf.is_tensor(data):
            data = data.numpy()
            
        scores = []
        for sample in data:  # (timesteps, features)
            for feature in range(data.shape[2]):
                series = sample[:, feature]
                if np.var(series) < 1e-6:  # Skip constant series
                    continue
                acf = np.correlate(series - np.mean(series), 
                                 series - np.mean(series), 
                                 mode='full')
                norm_acf = acf[len(series)-1 + lag] / (len(series) * np.var(series))
                scores.append(norm_acf)
        return np.nanmean(scores) if scores else 0.0
    
    def compute_wasserstein_distance(self, real_data: np.ndarray, synthetic_data: np.ndarray):
        """Compute Wasserstein Distance between real and synthetic distributions."""
        if tf.is_tensor(real_data):
            real_data = real_data.numpy()
        if tf.is_tensor(synthetic_data):
            synthetic_data = synthetic_data.numpy() 
            
        # Reshape data to 2D: (samples * time steps, features)
        real_flat = real_data.reshape(-1, real_data.shape[2])  # Shape: (samples * time steps, features)
        synthetic_flat = synthetic_data.reshape(-1, synthetic_data.shape[2])  # Shape: (samples * time steps, features)

        # Compute Wasserstein distance per feature
        wd_per_feature = [wasserstein_distance(real_flat[:, i], synthetic_flat[:, i]) for i in range(real_data.shape[2])]

        # Compute mean Wasserstein distance across all features
        mean_wd = np.mean(wd_per_feature)

        return mean_wd
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n == 0:
            # Generate synthetic data
            noise = tf.random.normal([self.real_data.shape[0], self.latent_dim])
            synthetic = self.model.generator(noise, training=False)
            
            # Calculate metrics
            #real_features = self.feature_extractor.predict(self.real_data)
            #fake_features = self.feature_extractor.predict(synthetic)

            #fid = self.calculate_fid(real_features, fake_features)
            fid = self.calculate_fid(self.real_data, synthetic)
            
            real_corr = self.temporal_correlation_score(self.real_data)
            temp_corr = self.temporal_correlation_score(synthetic)
            
            wasserstein_dist = self.compute_wasserstein_distance(self.real_data, synthetic)
            
            # Save best model
            if fid < self.best_fid:
                self.best_fid = fid
                self.model.generator.save_weights("best_generator_weights.h5")
            
            # Store and print
            self.fid_history.append(fid)
            self.temp_corr_history.append(temp_corr)
            self.real_corr_history.append(real_corr)
            self.wasserstein_dist_history.append(wasserstein_dist)
            # Print metrics
            print(f"\nEpoch {epoch+1} Metrics:")
            print(f"FID: {fid:.2f} (Lower better)")
            print(f"TempCorr: {temp_corr:.2f} (Real: {real_corr:.2f})")
            print(f"wasserstein_dist: {wasserstein_dist:.2f} (Lower better)")