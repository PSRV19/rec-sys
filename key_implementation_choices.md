# Explanation of Key Implementation Choices

- **Data Preprocessing:**

  - Ratings are converted to binary feedback (positive if rating ≥ 4).

  - Negative sampling is applied per positive interaction to generate balanced training data.

  - A train/validation/test split ensures that the model is evaluated on unseen data.

- **Model Architecture:**

  - Two separate embedding layers are used for the GMF and MLP branches to allow each branch to learn specialized representations.

  - The GMF branch uses element-wise multiplication while the MLP branch concatenates embeddings and passes them through several hidden layers.

  - A fusion layer concatenates both branch outputs before the final sigmoid activation to output a probability.

- **Training:**

  - Binary cross-entropy loss is chosen for binary classification.

  - Adam optimizer is used for its adaptive learning rate capabilities.

  - Early stopping monitors validation loss to prevent overfitting.

- **Evaluation:**

  - The evaluation computes Recall@10 and NDCG@10 by ranking candidate items for each user.

  - The evaluation evaluates and plots the model's performance for different top-k cutoffs.

  - The evaluation plots the training and loss curves to visualize training progress.

  - The approach ensures that only items unseen in training are considered for ranking.

This self-contained source code, with comprehensive inline comments, should help you understand and implement the NCF model.
