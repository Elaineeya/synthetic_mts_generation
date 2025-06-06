
import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns


# Load Data
def load_data(file_path):    
    df = pd.read_csv(file_path, sep=',', parse_dates=['timestamp'])
    return df


# prepare multivariate time series data for cwgangp models
def prepare_cwgangp_s_data(data, window_size=24, test_size=0.2, random_state=42):
    df = data.copy()
    # Convert to categorical features
    df['dayofweek'] = df['timestamp'].dt.day_name().str[:3]  # 7 categories
    df['month'] = df['timestamp'].dt.month_name().str[:3]  # 12 categories
    
    # Split data
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Get feature columns (excluding time info)
    countries = [col for col in df.columns if col not in ['timestamp','dayofweek','month']]
    
    # Scale numerical features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_df[countries])
    scaled_test = scaler.transform(test_df[countries])

    # Create encoder with fixed categories
    day_encoder = OneHotEncoder(categories=[['Mon','Tue','Wed','Thu','Fri','Sat','Sun']], sparse_output=False)
    month_encoder = OneHotEncoder(categories=[['Jan','Feb','Mar','Apr','May','Jun',
                                             'Jul','Aug','Sep','Oct','Nov','Dec']], sparse_output=False)
    
    # Fit on training data
    day_encoder.fit(train_df[['dayofweek']])
    month_encoder.fit(train_df[['month']])
    
    # Encode conditions for both splits
    def encode_conditions(df):
        day_encoded = day_encoder.transform(df[['dayofweek']])
        month_encoded = month_encoder.transform(df[['month']])
        return np.concatenate([day_encoded, month_encoded], axis=1)
    
    train_conditions = encode_conditions(train_df)
    test_conditions = encode_conditions(test_df)

    # Create sequences with original indices
    def create_sequences(data, conditions, original_indices):
        sequences = []
        condition_sequences = []
        position_indices = []
        for i in range(len(data) - window_size + 1):
            sequences.append(data[i:i+window_size])
            condition_sequences.append(conditions[i:i+window_size])
            position_indices.append(original_indices[i:i+window_size])
        return (
            np.array(sequences, dtype='float32'),
            np.array(condition_sequences, dtype='float32'),
            np.array(position_indices, dtype='int')
        )
    
    # Create sequences with original indices
    train_indices = np.arange(len(scaled_train))
    test_indices = np.arange(len(scaled_test))
    
    X_train, C_train, P_train = create_sequences(scaled_train, train_conditions, train_indices)
    X_test, C_test, P_test = create_sequences(scaled_test, test_conditions, test_indices)
    
    # Shuffle with index preservation
    def shuffle_sequences(X, C, P):
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(len(X))
        return X[indices], C[indices], P[indices]
    
    X_train, C_train, P_train = shuffle_sequences(X_train, C_train, P_train)
    X_test, C_test, P_test = shuffle_sequences(X_test, C_test, P_test)
    
    return scaler, (X_train, C_train, P_train), (X_test, C_test, P_test)

# prepare multivariate time series data for other generative models
def prepare_mts_s_data(data, window_size=24, test_size=0.2, random_state=42):
    df = data.copy()
    countries = df.columns[1:]  # Exclude timestamp
    
    # Split data
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Scale numerical features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_df[countries])
    scaled_test = scaler.transform(test_df[countries])

    # Create sequences with original indices
    def create_sequences(data, original_indices):
        sequences = []
        position_indices = []
        for i in range(len(data) - window_size + 1):
            sequences.append(data[i:i+window_size])
            position_indices.append(original_indices[i:i+window_size])
        return (
            np.array(sequences, dtype='float32'),
            np.array(position_indices, dtype='int')
        )
    
    # Create sequences with original indices
    train_indices = np.arange(len(scaled_train))
    test_indices = np.arange(len(scaled_test))

    
    X_train, P_train = create_sequences(scaled_train, train_indices)
    X_test, P_test = create_sequences(scaled_test, test_indices)
    
    # Shuffle with index preservation
    def shuffle_sequences(X, P):
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(len(X))
        return X[indices], P[indices]
    
    X_train, P_train = shuffle_sequences(X_train, P_train)
    X_test, P_test = shuffle_sequences(X_test, P_test)
    
    return scaler, (X_train, P_train), (X_test, P_test)


# Reconstruct synthetic time series data from generated windows
def reconstruct_synthetic_consecutive(scaler, synthetic_data, position_indices):
    """
    Reconstructs time series by:
    1. Selecting windows that fill consecutive segments
    2. Using the last valid window for remaining positions
    3. Applying inverse scaling
    """
     # Get components from prepared data
    samples, timesteps, features = synthetic_data.shape
    
    original_length = samples + timesteps - 1  # Adjust original length for reconstruction

    # 1. Inverse shuffle: Sort windows by their original temporal order
    # Get first position of each window to determine original order
    start_positions = position_indices[:, 0]  # Shape: (num_windows,)
    sort_idx = np.argsort(start_positions)  # Get sorting indices
    
    # Sort both synthetic data and position indices
    ordered_windows = synthetic_data[sort_idx]
    ordered_positions = position_indices[sort_idx]


    # 2. Reconstruct with consecutive window filling
    scaled_2d = np.zeros((original_length, features))
    current_pos = 0
    
    # First pass: Fill complete windows from start
    for window_idx in range(ordered_windows.shape[0]):
        window_start = ordered_positions[window_idx][0]
        
        # Check if this window aligns with current position
        if window_start == current_pos:
            # Fill the entire window
            end_pos = current_pos + timesteps
            if end_pos > original_length:
                break
                
            scaled_2d[current_pos:end_pos] = ordered_windows[window_idx]
            current_pos = end_pos
            
            if current_pos >= original_length:
                break

    # 3. Fill remaining positions with last valid window
    if current_pos < original_length:
        # Find the last window that contains remaining positions
        for window_idx in reversed(range(ordered_windows.shape[0])):
            window_start = ordered_positions[window_idx][0]
            window_end = ordered_positions[window_idx][-1]
            
            if window_end >= current_pos:
                # Calculate how many positions we need to fill
                fill_length = original_length - current_pos
                start_idx = current_pos - window_start
                
                scaled_2d[current_pos:] = ordered_windows[window_idx, start_idx:start_idx+fill_length]
                break

    # 4. Inverse scaling
    scaled_reconstructed = scaler.inverse_transform(scaled_2d)
    
    return scaled_reconstructed




# plot t-SNE for each feature across countries
def plot_feature_tsne(
    real_data: np.ndarray,  # Shape: (samples, 24, 21)
    synthetic_data: np.ndarray,
    country_names: list,
    max_samples: int = 500
):
    num_features = real_data.shape[2]
    n_cols = 3
    n_rows = int(np.ceil(num_features / n_cols))
    
    plt.figure(figsize=(18, 5*n_rows))
    
    for feature_idx in range(num_features):
        ax = plt.subplot(n_rows, n_cols, feature_idx+1)
        
        # Extract country data
        # num of samples used in the t-SNE plot
        used_samples = min(real_data.shape[0], max_samples)
        real_feature = real_data[:used_samples, :, feature_idx]
        synthetic_feature = synthetic_data[:used_samples, :, feature_idx]
        
        # Combine and reduce
        combined = np.vstack([real_feature, synthetic_feature])
        tsne = TSNE(n_components=2, perplexity=40, random_state=42)
        tsne_results = tsne.fit_transform(combined)
        
        # Plot
        ax.scatter(tsne_results[:used_samples, 0], tsne_results[:used_samples, 1], 
                   c='red', alpha=0.5, label='Real',s=100)
        ax.scatter(tsne_results[used_samples:, 0], tsne_results[used_samples:, 1], 
                   c='blue', alpha=0.5, label='Synthetic',s=100)
        ax.set_title(f"t-SNE for {country_names[feature_idx]}")
        ax.legend(fontsize='large')
    
    plt.tight_layout()
    plt.legend()
    plt.show()



# Assuming:
# - real_data and fake_data are numpy arrays or DataFrames of shape (559, 21)
# - column_names contains the 21 feature names (e.g., country names)
# plots real vs fake data for each feature in a grid layout
def plot_real_vs_fake(real_data, fake_data, column_names, figsize=(20, 25)):
    plt.figure(figsize=figsize)
    
    # Create subplots in a grid (7 rows x 3 columns for 21 features)
    n_rows, n_cols = 7, 3
    for idx in range(real_data.shape[1]):
        plt.subplot(n_rows, n_cols, idx+1)
        plt.plot(real_data[:, idx], label='Real', alpha=0.8, linewidth=1, color='red')
        plt.plot(fake_data[:, idx], label='Fake', alpha=0.8, linewidth=1, color='blue')
        plt.title(column_names[idx])
        plt.grid(alpha=0.3)
        plt.legend(fontsize='large')
            
    
    plt.tight_layout()
    plt.show()


# Plot training losses for VAE based models
def plot_vae_losses(model_history):
    plt.figure(figsize=(12, 6))
    
    # Plot Total Loss
    plt.subplot(1, 2, 1)
    plt.plot(model_history.history['total_loss'])
    plt.title('Training Total Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(True)
    
    # Plot Reconstruction Loss and KL Loss
    plt.subplot(1, 2, 2)
    plt.plot(model_history.history['recon_loss'], label='Reconstruction Loss')
    plt.plot(model_history.history['kl_loss'], label='KL Loss')
    plt.title('Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


# plot training metrics for VAE based models
def plot_vae_training_metrics(monitor_results):
    plt.figure(figsize=(12, 6))
    
    # FID Plot
    plt.subplot(1, 2, 1)
    plt.plot(monitor_results.fid_history, label='FID')
    plt.xlabel('Epoch (x10)')
    plt.ylabel('FID Score')
    plt.title('Frechet Inception Distance')
    plt.grid(True)
    
    # Temporal Correlation Plot
    plt.subplot(1, 2, 2)
    plt.plot(monitor_results.corr_history, 
             label='Synthetic Correlation', 
             color='orange')
    plt.xlabel('Epoch (x10)')
    plt.ylabel('Correlation Score')
    plt.title('Temporal Correlation')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


# Plot training losses for GAN based models
def plot_gan_losses(model_history):

    plt.figure(figsize=(12,6))
    plt.plot(model_history.history['c_loss'], label='Discriminator Loss')
    plt.plot(model_history.history['g_loss'], label='Generator Loss')
    plt.title('CWGAN-GP Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



# plot training metrics for VAE based models
def plot_gan_training_metrics(monitor_results):
    plt.figure(figsize=(12,6))

    # FID Plot
    plt.subplot(1,3,1)
    plt.plot(monitor_results.fid_history, label='FID')
    plt.xlabel('Epoch (x10)')
    plt.ylabel('FID Distance')
    plt.title('Frechet Inception Distance')
    plt.grid(True)

    # Temporal Correlation Plot
    plt.subplot(1,3,2)
    plt.plot(monitor_results.temp_corr_history, label='Synthetic', color='orange')
    plt.plot(monitor_results.real_corr_history, color='green', linestyle='--', label='Real Data')
    plt.xlabel('Epoch (x10)')
    plt.ylabel('Correlation Score')
    plt.title('Temporal Correlation')
    plt.legend()
    plt.grid(True)

    # Temporal Correlation Plot
    plt.subplot(1,3,3)
    plt.plot(monitor_results.wasserstein_dist_history, label='Wassertein Distances')
    plt.xlabel('Epoch (x10)')
    plt.ylabel('mean wassertein distances')
    plt.title('Wassertein Distances')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()