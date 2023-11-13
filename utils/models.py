import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class NeuralNetwork(nn.Module):
    def __init__(self, layers: int = 4, input_dimension: int = 1, output_dimension: int = 1, hidden_nodes = None, activation_function = None, softmax: bool = False):
        super().__init__()
        self.input_dimension = input_dimension
        self.hidden_layers = nn.ModuleList([nn.Linear(input_dimension, hidden_nodes, dtype=torch.float64)])
        self.hidden_layers.extend([nn.Linear(hidden_nodes, hidden_nodes, dtype=torch.float64) for _ in range(layers - 2)])
        self.output_layer = nn.Linear(hidden_nodes, output_dimension, dtype=torch.float64)
        self.activation_function = nn.ReLU() if activation_function is None else activation_function
        self.softmax = softmax
    
    def forward (self, x):
        for hidden_layer in self.hidden_layers:
            x = self.activation_function(hidden_layer(x))
        x = self.output_layer(x)
        if self.softmax:
            x = nn.Softmax(dim=1)(x)
        return x
    
    def train_model(self, x, y, epochs: int = 200, batch_size: int = 32, loss_function = None, optimizer = None):
        x_tensor, y_tensor = x, y
        if not type(x_tensor) == torch.Tensor:
            x_tensor, y_tensor = torch.tensor(x, dtype=torch.float64).view(-1, self.input_dimension), torch.tensor(y, dtype=torch.float64).view(-1)
        dataset = TensorDataset(x_tensor, y_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Define the loss function and optimizer. Use cross entropy loss instead of MSE to accomodate 2 output params
        criterion = nn.CrossEntropyLoss() if loss_function == None else loss_function
        optimizer = optim.Adam(self.parameters()) if optimizer == None else optimizer(self.parameters(), lr = 0.1) 

        # Training loop
        for epoch in range(1,epochs):
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
            x_tensor = torch.tensor(x, dtype=torch.float64).view(-1, self.input_dimension)
        with torch.no_grad():  # Disable gradient computation
            y_pred = self(x_tensor)
            return (y_pred)


            
            
