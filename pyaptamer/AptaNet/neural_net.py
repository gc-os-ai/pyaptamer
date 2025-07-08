import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


class MLP(nn.Module):
    """
    Multi-layer Perceptron (MLP) for binary classification with feature selection.

    This model uses a RandomForest-based feature selector and a deep neural network
    with AlphaDropout for robust training.

    Parameters
    ----------
    input_dim : int, optional
        Number of input features before feature selection (default is 639).

    Attributes
    ----------
    clf : RandomForestClassifier
        Random forest used for feature selection.
    clf_model : SelectFromModel or None
        Feature selector model after fitting.
    criterion : nn.BCELoss
        Binary cross-entropy loss function.
    optimizer : torch.optim.Optimizer or None
        Optimizer for training.
    """

    def __init__(self, input_dim=639):
        """
        Initialize the MLP model.

        Parameters
        ----------
        input_dim : int, optional
            Number of input features before feature selection (default is 639).
        """
        super(MLP, self).__init__()

        self.input_dim = input_dim

        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.AlphaDropout(0.3)
        self.fc2 = nn.Linear(128, 128)
        self.dropout2 = nn.AlphaDropout(0.3)
        self.fc3 = nn.Linear(128, 128)
        self.dropout3 = nn.AlphaDropout(0.3)
        self.fc4 = nn.Linear(128, 128)
        self.dropout4 = nn.AlphaDropout(0.3)
        self.fc5 = nn.Linear(128, 128)
        self.dropout5 = nn.AlphaDropout(0.3)
        self.fc6 = nn.Linear(128, 128)
        self.dropout6 = nn.AlphaDropout(0.3)
        self.fc7 = nn.Linear(128, 1)  # Binary classification

        self.criterion = nn.BCELoss()
        self.optimizer = None

        # Random forest feature selector
        self.clf = RandomForestClassifier(n_estimators=300, max_depth=9, random_state=0)
        self.clf_model = None

    def _forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_features).

        Returns
        -------
        torch.Tensor
            Output tensor with predicted probabilities.
        """
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)
        x = F.relu(self.fc5(x))
        x = self.dropout5(x)
        x = F.relu(self.fc6(x))
        x = self.dropout6(x)
        x = torch.sigmoid(self.fc7(x))
        return x

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

    def _transform_X(self, X):
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
        Fit the MLP model to the training data.

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
        # Convert to PyTorch tensors
        self.clf_model = self._train_clf_model(X, y)
        X_transformed = self._transform_X(X)
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
        for epoch in range(epochs):
            for batch_x, batch_y in dataloader:
                self.optimizer.zero_grad()
                output = self._forward(batch_x)
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
        X_transformed = self.transform_X(X)
        X_tensor = torch.tensor(X_transformed, dtype=torch.float32)
        with torch.no_grad():
            output = self._forward(X_tensor)
            predictions = (output > 0.5).int().squeeze().numpy()
        return predictions
