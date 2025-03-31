# Recommender Systems A1
Group 33 Recommender Systems

# Neural Collaborative Filtering (NCF) Implementation

This repository contains an implementation of the Neural Collaborative Filtering (NCF) model using PyTorch on the MovieLens 1M dataset. The model combines matrix factorization (GMF) and a Multi-Layer Perceptron (MLP) to predict user-item interactions.

## Repository Structure

- **ncf.py**: Main Python source file that includes data preprocessing, model definition, training, and evaluation.
- **ratings.dat**: (Not included) Download the MovieLens 1M dataset from the [MovieLens Official Website](https://grouplens.org/datasets/movielens/1m/) and place the `ratings.dat` file in the same directory.
- **README.md**: This file.

## Requirements

- Python 3.6+
- PyTorch (version 1.6+ recommended)
- pandas
- numpy
- scikit-learn

You can install the required packages using pip:

```bash
pip install torch pandas numpy scikit-learn
```

## Getting Started

1. **Download the MovieLens 1M Dataset**

    Visit the [MovieLens Official Website](https://grouplens.org/datasets/movielens/1m/) and download the dataset. Extract the file and locate `ratings.dat`.

    Place ratings.dat in the same directory as ncf.py.

2. **Run the Code**

    Open a terminal in the project directory and execute the following command:

    ``` bash
    python ncf.py
    ```

    The script will:
    - Load and preprocess the MovieLens data.
    - Convert explicit ratings into binary feedback (positive interactions for ratings ≥ 4).
    - Generate negative samples.
    - Split the data into training (70%), validation (15%), and testing (15%) sets.
    - Build and train the NCF model.
    - Evaluate the model using Recall@10 and NDCG@10 metrics, then print the results.

## Code Overview

- **Data Preprocessing**

    The code reads `ratings.dat`, converts ratings to binary labels, performs negative sampling, and splits the dataset into training, validation, and test sets.

- **Model Architecture**

    The NCF model is composed of:

  - Two embedding layers for each of the GMF and MLP branches.

  - An MLP that processes concatenated user and item embeddings.

  - A fusion layer that combines outputs from both branches before applying a sigmoid activation for the final prediction.

- **Training and Evaluation**

    The model is trained using binary cross-entropy loss and the Adam optimizer. Early stopping is implemented based on validation loss. Evaluation is performed using Recall@10 and NDCG@10 metrics.

## Customization

- **Hyperparameters:**

  - You can adjust hyperparameters (e.g., embedding dimension, learning rate, number of MLP layers) directly in the `ncf.py` file.

- **Negative Sampling:**

    The number of negative samples per positive example is controlled by the `num_negatives` parameter in the `load_and_preprocess_data` function.

## Troubleshooting

- **Dataset Not Found:**

    Ensure that the `ratings.dat` file is in the same directory as `ncf.py` and that its path is correctly set in the code.

- **Dependency Issues:**

    Verify that all required Python packages are installed. You can install any missing packages using pip.

## Additional Notes

- The code sets random seeds for reproducibility, but results may vary slightly due to nondeterministic behavior in some PyTorch operations.
