import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


class MLP(nn.Module):
    def __init__(self, input_dim=639):
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

    def forward(self, x):
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

    def train_clf_model(self, X, y):
        self.clf.fit(X, y)
        self.clf_model = SelectFromModel(self.clf, prefit=True)
        return self.clf_model

    def transform_X(self, X):
        if self.clf_model is None:
            raise ValueError("Classifier model has not been trained yet.")
        return self.clf_model.transform(X)

    def fit_model(self, X, y, epochs=200, batch_size=310):
        # Convert to PyTorch tensors
        self.clf_model = self.train_clf_model(X, y)
        X_transformed = self.transform_X(X)
        X_tensor = torch.tensor(X_transformed, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        self.optimizer = optim.RMSprop(self.parameters(), lr=0.00014, alpha=0.9, eps=1e-07)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.train()
        for epoch in range(epochs):
            for batch_x, batch_y in dataloader:
                self.optimizer.zero_grad()
                output = self.forward(batch_x)
                loss = self.criterion(output, batch_y)
                loss.backward()
                self.optimizer.step()

    def predict(self, X):
        self.eval()
        X_transformed = self.transform_X(X)
        X_tensor = torch.tensor(X_transformed, dtype=torch.float32)
        with torch.no_grad():
            output = self.forward(X_tensor)
            predictions = (output > 0.5).int().squeeze().numpy()
        return predictions