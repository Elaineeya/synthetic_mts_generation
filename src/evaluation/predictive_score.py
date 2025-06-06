#%% [markdown]
### Predictive Metrics
### predicts all features (i.e., the full vector at the next time step)
### (i.e., electricity prices for all countries at the next time step)
### The MAE is computed across all features, not just the last one.
#%%
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, TimeDistributed
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.optimizers import Adam


def all_feature_predictive_score_metrics(ori_data, generated_data):
    """Updated for TF2 eager execution and multivariate prediction"""
    # Convert to numpy arrays if needed
    ori_data = np.array(ori_data).astype(np.float32)
    generated_data = np.array(generated_data).astype(np.float32)
    
    # Parameters
    n_samples, seq_len, n_features = ori_data.shape
    lookback = seq_len - 1  # Predict next step in sequence
    
    # Prepare training data from synthetic data
    X_train = generated_data[:, :-1, :]  # All features except last time step
    y_train = generated_data[:, 1:, :]   # Shift one step forward for all features
    
    # Build and train model
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(lookback, n_features)),
        Dense(n_features) # Predict all features at each time step
        #TimeDistributed(Dense(n_features))  
    ])
    
    model.compile(optimizer='adam', loss='mae')
    model.fit(X_train, y_train, epochs=50, batch_size=128, verbose=0)
    
    # Prepare test data from real data
    X_test = ori_data[:, :-1, :]
    y_test = ori_data[:, 1:, :]
    
    # Evaluate
    test_pred = model.predict(X_test, verbose=0)
    #mae_score = mean_absolute_error(y_test.reshape(-1, n_features), 
    #                           test_pred.reshape(-1, n_features))
    
    # Compute MAE across all features
    MAE_scores = [mean_absolute_error(y_test[i].flatten(), test_pred[i].flatten()) for i in range(n_samples)]
    predictive_score = np.mean(MAE_scores)
    
    return predictive_score
    #return mae_score


#### Yoon's TimeGAN paper about prediction scores

##Yoon’s method trains a GRU-based RNN on the synthetic dataset to predict the last feature (dim-1) at each time step, 
# and then evaluates it on the real dataset. 
## The Mean Absolute Error (MAE) between the predictions and actual values serves as the performance metric.


def extract_time (data):
    """Returns Maximum sequence length and each sequence length.
  
    Args:
       - data: original data
    
    Returns:
       - time: extracted time information
       - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:,0]))
        time.append(len(data[i][:,0]))
    return time, max_seq_len

def build_predictor(hidden_dim, max_seq_len, feature_dim):
    """Build a simple GRU-based predictor model."""
    model = tf.keras.Sequential([
        GRU(hidden_dim, activation='tanh', return_sequences=True, input_shape=(max_seq_len - 1, feature_dim - 1)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='mae')
    return model

def last_feature_predictive_score_metrics(ori_data, generated_data):
    """Evaluate synthetic data quality using post-hoc RNN one-step ahead prediction."""
    # Extract time information
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max(ori_max_seq_len, generated_max_seq_len)

    # Dataset dimensions
    no, seq_len, feature_dim = np.asarray(ori_data).shape
    hidden_dim = feature_dim // 2
    batch_size = 128
    epochs = 50  # Reduced for efficiency (original used 5000 iterations)

    # Create predictor model
    predictor = build_predictor(hidden_dim, max_seq_len, feature_dim)

    # Compile model
    predictor.compile(optimizer=Adam(), loss='mae')

    # Prepare training data from synthetic dataset
    X_train = np.array([generated_data[i][:-1, :(feature_dim - 1)] for i in range(len(generated_data))])
    Y_train = np.array([generated_data[i][1:, (feature_dim - 1)] for i in range(len(generated_data))])
    Y_train = Y_train.reshape(Y_train.shape[0], Y_train.shape[1], 1)  # Ensure correct shape

    # Train predictor on synthetic data
    predictor.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Prepare test data from real dataset
    X_test = np.array([ori_data[i][:-1, :(feature_dim - 1)] for i in range(len(ori_data))])
    Y_test = np.array([ori_data[i][1:, (feature_dim - 1)] for i in range(len(ori_data))])
    Y_test = Y_test.reshape(Y_test.shape[0], Y_test.shape[1], 1)  # Ensure correct shape

    # Make predictions on real data
    Y_pred = predictor.predict(X_test)

    # Compute MAE between predictions and real data
    MAE_scores = [mean_absolute_error(Y_test[i], Y_pred[i]) for i in range(no)]
    predictive_score = np.mean(MAE_scores)

    return predictive_score




# Create masked datasets
def prepare_data(data, target_idx):
    # Remove target feature from input
    X = np.delete(data, target_idx, axis=-1)[:, :-1, :]
    # Target is next steps of removed feature
    Y = data[:, 1:, target_idx, np.newaxis]  # Add channel dimension
    return X, Y


def feature_wise_predictive_score_metrics(ori_data, generated_data):
    """Enhanced version that predicts each feature using others and returns MAE scores
    
    Args:
        ori_data: Real data (n_samples, seq_len, n_features)
        generated_data: Synthetic data (n_samples, seq_len, n_features)
        
    Returns:
        dict: MAE scores for each feature and average score
    """
    # Extract time information (assuming fixed-length sequences)
    ori_time, ori_max_seq_len = extract_time(ori_data)
    generated_time, generated_max_seq_len = extract_time(generated_data)
    max_seq_len = max(ori_max_seq_len, generated_max_seq_len)
    
    # Dataset parameters
    n_samples, seq_len, feature_dim = ori_data.shape
    hidden_dim = feature_dim // 2
    batch_size = 128
    epochs = 50
    
    # Dictionary to store results
    mae_scores = {}
    
    # Create models and evaluate for each feature
    for feature_idx in range(feature_dim):
        #print(f"Processing feature {feature_idx+1}/{feature_dim}")
        
        # Prepare training data (synthetic)
        X_train, Y_train = prepare_data(generated_data, feature_idx)
        
        # Prepare test data (real)
        X_test, Y_test = prepare_data(ori_data, feature_idx)
        
        # Train model
        predictor = build_predictor(hidden_dim, max_seq_len, feature_dim)
        predictor.fit(X_train, Y_train, 
                     epochs=epochs, 
                     batch_size=batch_size, 
                     verbose=0)
        
        # Evaluate on real data
        Y_pred = predictor.predict(X_test, verbose=0)
        mae = mean_absolute_error(Y_test.flatten(), Y_pred.flatten())
        mae_scores[f'feature_{feature_idx}'] = mae
    
    # Calculate average MAE
    avg_mae = np.mean(list(mae_scores.values()))
    
    return avg_mae



