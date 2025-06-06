import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class DiscriminativeScore(tf.keras.Model):
    """Discriminator Network for Time-series Data"""
    def __init__(self, seq_len, num_features):
        super().__init__()
        gru_units = max(1, int(num_features/2))
        self.gru = tf.keras.layers.GRU(
            units=gru_units, 
            activation='tanh',
            return_sequences=False
        )
        self.classifier = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x, seq_lens = inputs
        output = self.gru(x, mask=tf.sequence_mask(seq_lens))
        return self.classifier(output)

def discriminative_score(real_data, synthetic_data):
    """
    Compute discriminative score for multivariate time-series data
    
    Args:
        real_data: np.array of shape (n_samples, seq_len, n_features)
        synthetic_data: np.array of same shape as real_data
    
    Returns:
        discriminative_score: float between 0-0.5 (lower is better)
    """
    # Data preparation
    n_samples = real_data.shape[0]
    seq_len = real_data.shape[1]
    n_features = real_data.shape[2]
    
    # Create labels: 1 for real, 0 for synthetic
    X = np.concatenate([real_data, synthetic_data], axis=0)
    y = np.concatenate([np.ones(n_samples), np.zeros(n_samples)], axis=0)
    
    # Create sequence lengths (assuming fixed-length sequences)
    seq_lens = np.full(X.shape[0], seq_len)
    
    # Train-test split
    X_train, X_test, y_train, y_test, sl_train, sl_test = train_test_split(
        X, y, seq_lens, test_size=0.2, stratify=y
    )
    
    # Build and compile discriminator
    discriminator = DiscriminativeScore(seq_len, n_features)
    discriminator.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train discriminator
    discriminator.fit(
        x=(X_train, sl_train),
        y=y_train,
        epochs=50,
        batch_size=128,
        validation_split=0.1,
        verbose=0
    )
    
    # Evaluate
    _, test_acc = discriminator.evaluate(
        (X_test, sl_test), y_test, verbose=0)
    
    return np.abs(0.5 - test_acc)





def countrywise_analysis(real_data, synthetic_data):
    """Main analysis function with country-wise comparisons"""
    # Compute multivariate score
    mv_score = discriminative_score(real_data, synthetic_data)
    print(f"\nMultivariate Discriminative Score: {mv_score:.4f}")

    # Compute country-wise scores
    num_features = real_data.shape[2]
    country_scores = []
    
    for country_idx in range(num_features):
        # Extract single country data and add channel dimension
        real_country = real_data[:, :, country_idx][..., np.newaxis]
        synthetic_country = synthetic_data[:, :, country_idx][..., np.newaxis]
        
        # Compute country score
        country_score = discriminative_score(real_country, synthetic_country)
        country_scores.append(country_score)
        
        # Print comparison
        comparison = "HIGHER" if country_score > mv_score else "lower"
        print(f"Country {country_idx+1} Score: {country_score:.4f} ({comparison} than multivariate)")

    # Identify problematic countries
    threshold = mv_score * 1.2  # 120% higher than multivariate score
    problematic_countries = np.where(np.array(country_scores) > threshold)[0]
    
    print("\nAnalysis Results:")
    print(f"Multivariate Score: {mv_score:.4f}")
    print(f"Threshold for Problematic Countries: {threshold:.4f}")
    print("Countries contributing significantly to discriminability:")
    for idx in problematic_countries:
        print(f"  Country {idx+1}: {country_scores[idx]:.4f}")

    return mv_score, country_scores


def plot_country_wise_discriminative_scores(mv_score, country_scores, country_names):

    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Create country indices and labels
    countries = np.arange(1, len(country_scores) + 1)
    
    # Plot country scores as bars
    bars = plt.bar(countries, country_scores, 
                 color=['red' if score > mv_score else 'steelblue' 
                        for score in country_scores])
    
    # Add multivariate score line
    plt.axhline(y=mv_score, color='green', linestyle='--', 
              linewidth=2, label='Multivariate Score')
    
    # Style the plot
    plt.title("Country-wise Discriminative Scores vs Multivariate Baseline", fontsize=14)
    plt.xlabel("Countries", fontsize=12)
    plt.ylabel("Discriminative Score", fontsize=12)
    plt.xticks(countries, country_names, rotation=45, ha='right')
    plt.legend()
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
