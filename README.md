
# 📈 Synthetic Data Generation for Energy Markets Using Generative Models

This repository contains the complete codebase for my master’s thesis project conducted in collaboration with **Siemens**. The goal of this project is to generate realistic **synthetic electricity load and price time-series data** using generative models and evaluate their effectiveness for downstream forecasting and classification tasks.

---

## 🧠 Project Overview

Real-world electricity data often suffers from limitations like **small data sizes**, **noise**, and **missing values**—especially for rare events. This project investigates the use of **Variational Autoencoders (VAEs)** and **Generative Adversarial Networks (GANs)** to generate synthetic, multivariate time-series data based on hourly electricity price and load data from **ENTSO-E**.

---

## 🗂️ Repository Structure

```
.
├── data/
├── evaluation/
├── utilities/
├── data_utils.py
├── gan.py
├── vae.py
├── vae_timevae_price_2.ipynb
├── wgan_cwgan_price_2.ipynb
├── quantitative_evaluation.ipynb
└── README.md
```

### 📁 `data/`
This folder contains all the **preprocessed and sampled datasets** used throughout the thesis project. The data includes hourly electricity **price and load data** for multiple European countries from the ENTSO-E platform.

---

### 📁 `utilities/`

This folder provides utility scripts for **data extraction** and **initial preprocessing**:

- `entsoe_sample_extraction.py`:  
  Sample code to extract raw electricity data from the ENTSO-E platform using their API.

- `exploration_and_preprocessing.py`:  
  Includes:
  - **Exploratory analysis**: Generates country-wise heatmaps to visualize electricity trends over time.
  - **Basic preprocessing**: Handles missing data and generates datasets with various sample sizes (2%, 10%, 20%, 50%, 100%).

---

### 🧾 `data_utils.py`

This file contains essential helper functions used across the notebooks and models:

- `load_data()`: Load cleaned and prepared electricity datasets.
- `prepare_mts_s_data() and prepare_cwgangp_s_data()`: Normalize and window the data for training.
- `plot_feature_tsne()`: t-SNE visualization of real vs. synthetic data in latent space.
- `plot_real_vs_fake()`: Visual comparison of real and synthetic sequences.

---

### 🧾 `gan.py` and `vae.py`

These scripts contain implementations of the **four generative models** evaluated in the thesis:

- **VAE** (variatioonal Auutooencoder)
- **TimeVAE** (decomposes time series into trend, seasonal, and level components)
- **WGAN-GP** (Wasserstein GAN with gradient penalty)
- **CWGAN-GP** (Conditional WGAN-GP using day-of-week and month as conditioning variables)

---

### 📁 `evaluation/`

Contains code for **quantitative evaluation** of synthetic data:

- `predictive_score.py`:  
  Evaluates how well synthetic data preserves temporal dependencies by training a GRU model to predict future values.

- `discriminative_score.py`:  
  Measures whether a classifier can distinguish real data from synthetic data.

- `classification_score.py`:  
  Evaluates whether synthetic data can eeffectively suppoort downstream classification tasks, such as classifying market behavior.

---

### 📒 `vae_timevae_price_2.ipynb`

An interactive notebook that demonstrates:

- How to load and preprocess data.
- How to build, train, and evaluate the **VAE** and **TimeVAE** models.
- How to generate synthetic time-series samples and visualize results.

---

### 📒 `wgan_cwgan_price_2.ipynb`

An interactive notebook that shows:

- How to load and preprocess data.
- How to build, train, and evaluate the **WGAN-GP** and **CWGAN-GP** models.
- How to generate synthetic data samples and visualize results.

---

### 📒 `quantitative_evaluation.ipynb`

A complete walkthrough of how to:

- Evaluate all four models using the **predictive**, **discriminative**, and **classification** scores.
- Compare model performance quantitatively.

---

## 📌 How to Run

1. Clone this repository.
2. Set up your Python environment (Python 3.10.9 is used in this project).
3. Install required libraries (See requirements.txt).
4. Start with the example notebooks:
   - `vae_timevae_price_2.ipynb`
   - `wgan_cwgan_price_2.ipynb`
   - `quantitative_evaluation.ipynb`
