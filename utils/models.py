import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.regularizers import l2
torch.set_default_dtype(torch.float32)


class pytorch_nn(nn.Module):
    def __init__(self, hidden_layers: int = 2, input_dimension: int = 1, output_dimension: int = 1, hidden_nodes=None, activation_function=None, softmax: bool = False):
        super().__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(input_dimension, hidden_nodes)])
        self.hidden_layers.extend([nn.Linear(
            hidden_nodes, hidden_nodes) for _ in range(hidden_layers)])
        self.output_layer = nn.Linear(
            hidden_nodes, output_dimension)
        self.activation_function = nn.ReLU(
        ) if activation_function is None else activation_function
        self.softmax = softmax

    def forward(self, x):
        for hidden_layer in self.hidden_layers:
            x = self.activation_function(hidden_layer(x))
        x = self.output_layer(x)
        if self.softmax:
            x = nn.Softmax(dim=1)(x)
        return x

    def train_model(self, x, y, epochs: int = 200, batch_size: int = 32, loss_function=None, optimizer=None):
        x_tensor, y_tensor = x, y
        if not type(x_tensor) == torch.Tensor:
            x_tensor = torch.tensor(
                x, dtype=torch.float32).view(-1, self.input_dimension)
            y_tensor = torch.tensor(
                y, dtype=torch.float32).view(-1, self.output_dimension)
        dataset = TensorDataset(x_tensor, y_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Define the loss function and optimizer. Use cross entropy loss instead of MSE when doing discrete classification
        criterion = nn.CrossEntropyLoss() if loss_function == None else loss_function
        optimizer = optim.Adam(self.parameters()) if optimizer == None else optimizer(
            self.parameters(), lr=0.1)

        # Training loop
        for epoch in range(1, epochs):
            for batch in data_loader:
                batch_x, batch_y = batch
                # Forward pass
                y_pred = self(batch_x)
                loss = criterion(y_pred, batch_y)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print statistics
            if (epoch) % 10 == 0:
                print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item()}')

    def predict(self, x):
        self.eval()
        x_tensor = x
        if not type(x_tensor) == torch.Tensor:
            x_tensor = torch.tensor(
                x, dtype=torch.float32).view(-1, self.input_dimension)
        with torch.no_grad():  # Disable gradient computation
            y_pred = self(x_tensor)
            return (y_pred)


def keras_nn(input_dim=2, output_dim=2, hidden_layers=2, activation='relu', hidden_nodes=4, final_activation='softmax', l2_lambda=0.01):
    model = Sequential()
    model.add(Dense(units=hidden_nodes, activation=activation,
              input_dim=input_dim, kernel_regularizer=l2(l2_lambda)))
    for _ in range(hidden_layers):
        model.add(Dense(units=hidden_nodes, activation=activation,
                  kernel_regularizer=l2(l2_lambda)))
    model.add(Dense(units=output_dim, activation=final_activation,
              kernel_regularizer=l2(l2_lambda)))
    return (model)
