import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from pyaptamer.pseaac import PSeAAC
from pyaptamer.utils._aptanet_utils import generate_kmer_vecs


def aptanet_layer(input_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    """Create a single AptaNet layer with AlphaDropout and ReLU activation."""
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.ReLU(),
        nn.AlphaDropout(dropout),
    )


class AptaNet(nn.Module):
    """
    AptaNet deep neural network for classification and feature generation.

    This class generates a combined feature vector using k-mer frequencies from the
    aptamer DNA sequence and PseAAC features from the protein sequence, and provides
    a deep neural network to classify binding interactions between aptamers and
    proteins, with a binary output (0 or 1).

    References
    ----------
    Emami, N., Ferdousi, R. AptaNet as a deep learning approach for aptamer–protein
    interaction prediction. Sci Rep 11, 6074 (2021).

    Parameters
    ----------
    n_layers : int
        Number of hidden layers.
    hidden_dim : int
        Hidden layer dimension.
    output_dim : int
        Output feature dimension.
    dropout : float
        Dropout rate for AlphaDropout.

    Example
    --------
    >>> aptanet = AptaNet(n_layers=5, hidden_dim=128, dropout=0.3)
    >>> aptanet.fit(X_train, y_train, epochs=100, batch_size=32)
    >>> predictions = aptanet.predict(X_test)
    >>> feature_vector = aptanet.preprocessing(aptamer_sequence, protein_sequence)
    >>> print(feature_vector[:10])
    np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

    Methods
    ----------
    forward(X)
        Forward pass through the network.
    fit(X, y, epochs, batch_size)
        Fit the AptaNet model to the training data.
    predict(X)
        Predict binary class labels for input samples.
    preprocessing(aptamer_sequence, protein_sequence)
        Generate the final feature vector by concatenating k-mer and PseAAC features.
    """

    def __init__(
        self,
        n_layers=7,
        hidden_dim=128,
        dropout=0.3,
        output_dim=1,
    ):
        super().__init__()
        assert n_layers > 0, "Number of hidden layers must be greater than 0."

        self.input_dim = None  # Will be set after feature selection
        self.output_dim = output_dim  # Binary classification
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.criterion = nn.BCEWithLogitsLoss()
        # Optimizer will be initialized during training (lack of self.parameters())
        self.optimizer = None
        self.clf = RandomForestClassifier(
            n_estimators=300, max_depth=9, random_state=0
        )  # Random forest feature selector
        self.clf_model = None  # Feature selector model after fitting
        self.model = None  # Will be initialized after feature selection

    def _init_model(self, input_dim):
        """Initialize AptaNet's deep neural network after feature selection."""
        model = [aptanet_layer(input_dim, self.hidden_dim, self.dropout)]
        for _ in range(self.n_layers):
            model.append(aptanet_layer(self.hidden_dim, self.hidden_dim, self.dropout))
        model.append(nn.Linear(self.hidden_dim, self.output_dim))
        return nn.Sequential(*model)

    def forward(self, X):
        """
        Forward pass through the network.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (batch_size, n_features).

        Returns
        -------
        torch.Tensor
            Output tensor with predicted probabilities.
        """
        return self.model(X)

    def _train_clf_model(self, X, y):
        """
        Train the random forest feature selector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        SelectFromModel
            Trained feature selector.
        """
        self.clf.fit(X, y)
        self.clf_model = SelectFromModel(self.clf, prefit=True)
        return self.clf_model

    def _transform_x(self, X):
        """
        Transform input features using the trained feature selector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        array-like
            Transformed input samples with selected features.

        Raises
        ------
        ValueError
            If the feature selector has not been trained.
        """
        if self.clf_model is None:
            raise ValueError("Classifier model has not been trained yet.")
        return self.clf_model.transform(X)

    def fit(self, X, y, epochs=200, batch_size=310):
        """
        Fit the AptaNet model to the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.
        y : array-like of shape (n_samples,)
            Target values.
        epochs : int, optional
            Number of training epochs (default is 200).
        batch_size : int, optional
            Batch size for training (default is 310).
        """
        # Feature selection
        self.clf_model = self._train_clf_model(X, y)
        X_transformed = self._transform_x(X)
        self.input_dim = X_transformed.shape[1]
        # Initialize the model with the selected input dimension
        self.model = self._init_model(self.input_dim)
        X_tensor = torch.tensor(X_transformed, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        self.optimizer = optim.RMSprop(
            self.parameters(), lr=0.00014, alpha=0.9, eps=1e-07
        )

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        self.train()
        for _epoch in range(epochs):
            for batch_x, batch_y in dataloader:
                self.optimizer.zero_grad()
                output = self(batch_x)
                loss = self.criterion(output, batch_y)
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        """
        Predict binary class labels for input samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        numpy.ndarray
            Predicted class labels (0 or 1).
        """
        self.eval()
        X_transformed = self._transform_x(X)
        X_tensor = torch.tensor(X_transformed, dtype=torch.float32)
        with torch.no_grad():
            logits = self(X_tensor)
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).int().squeeze().numpy()
        return predictions

    def preprocessing(self, aptamer_sequence, protein_sequence):
        """
        Generate the final feature vector by concatenating k-mer and PseAAC features.

        Parameters
        ----------
        aptamer_sequence : str
            The DNA sequence of the aptamer.
        protein_sequence : str
            The protein sequence to be analyzed.

        Returns
        -------
        np.ndarray
            1D numpy array of combined feature vector for the aptamer-protein pair.
        """
        self.aptamer_sequence = aptamer_sequence
        self.protein_sequence = protein_sequence

        # Initialize the PseAAC object with the protein sequence
        self.pseaac = PSeAAC.transform(self.protein_sequence)

        final_vector = np.concatenate(
            [generate_kmer_vecs(self.aptamer_sequence, k=4), np.array(self.pseaac)]
        )

        return final_vector
