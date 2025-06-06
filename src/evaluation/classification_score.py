#%% [markdown]
### 2. Feature Classification Evaluation
#%%
from tensorflow.keras import Sequential, layers
from sklearn.model_selection import train_test_split
import numpy as np
def transform_for_classification(data):
    """
    Correct transformation for feature-wise classification
    Converts (num_samples, window_size, num_features) to:
    - Data: (num_samples*num_features, window_size, 1)
    - Labels: (num_samples*num_features,) with feature indices
    """
    num_samples, window_size, num_features = data.shape
    
    # Reshape data to (num_samples*num_features, window_size, 1)
    # For each sample, stack features as separate sequences
    reshaped_data = data.transpose(0, 2, 1)  # (samples, features, time)
    reshaped_data = reshaped_data.reshape(-1, window_size, 1)  # (samples*features, time, 1)
    
    # Create labels repeated for each sample
    labels = np.tile(np.arange(num_features), num_samples)
    
    return reshaped_data, labels

def build_classification_model(input_shape, num_classes):
    """CNN-based feature classifier"""
    model = Sequential([
        layers.Conv1D(64, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                 optimizer='adam',
                 metrics=['accuracy'])
    return model




def classification_score(real_data, synthetic_data, num_classes=21):
    """
    Builds, trains, and evaluates a CNN classifier.
    
    Args:
        synthetic_data: Synthetic data
        real_data: Real input data
        num_classes: Number of output classes (int)

    Returns:
        classification_error: float
    """

    X_syn_cls, y_syn_cls = transform_for_classification(synthetic_data)
    X_real_cls, y_real_cls = transform_for_classification(real_data)

    # Split synthetic data for training
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X_syn_cls, y_syn_cls, test_size=0.2, stratify=y_syn_cls)

    # Define classifier model
    model = Sequential([
        layers.Conv1D(64, 3, activation='relu', input_shape=(X_train_cls.shape[1], 1)),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Train on synthetic data
    model.fit(X_train_cls, y_train_cls,
              epochs=50,
              batch_size=256,
              validation_split=0.2,
              verbose=0)

    # Evaluate on real data
    real_loss, real_acc = model.evaluate(X_real_cls, y_real_cls, verbose=0)

    classification_error = 1 - real_acc
    return classification_error
