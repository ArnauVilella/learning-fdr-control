import torch
import torch.nn as nn

class FDPNet(nn.Module):
    def __init__(self, phi_shape):
        """
        Initialize the FDP prediction network.
        
        Parameters
        ----------
        phi_shape : tuple
            Shape of the Phi matrix (rows, columns)
        """
        super(FDPNet, self).__init__()
        self.phi_shape = phi_shape
        
        # Calculate input size: flattened Phi matrix + v + T_stop + L
        self.input_size = phi_shape[0] * phi_shape[1] + 3
        
        # Define the network layers
        self.fc1 = nn.Linear(self.input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        #self.fc1 = nn.Linear(self.input_size, 64)
        #self.fc2 = nn.Linear(64, 32)
        #self.fc3 = nn.Linear(32, 8)
        #self.fc4 = nn.Linear(8, 1)
        
    def forward(self, Phi, v, T_stop, L):
        """
        Forward pass through the network.
        
        Parameters
        ----------
        Phi : torch.Tensor
            Phi matrix of shape (batch_size, rows, columns)
        v : torch.Tensor
            Threshold value of shape (batch_size,)
        T_stop : torch.Tensor
            T_stop value of shape (batch_size,)
        L : torch.Tensor
            Number of dummies of shape (batch_size,)
            
        Returns
        -------
        torch.Tensor
            Predicted FDP value
        """
        batch_size = Phi.shape[0]
        
        # Flatten the Phi matrix
        Phi_flat = Phi.reshape(batch_size, -1)
        
        # Prepare v, T_stop and L for concatenation
        v = v.unsqueeze(1)
        T_stop = T_stop.unsqueeze(1)
        L = L.unsqueeze(1)
        
        # Concatenate the flattened Phi matrix, v, T_stop and L
        x = torch.cat((Phi_flat, v, T_stop, L), dim=1)
        
        # Pass through the network
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        
        return x


class FDPNetCNN(nn.Module):
    def __init__(self, phi_shape):
        """
        Initialize the FDP prediction network with a CNN encoder for temporal data.

        Parameters
        ----------
        phi_shape : tuple
            Shape of the Phi matrix (p, T) where p is features and T is time steps.
        """
        super(FDPNetCNN, self).__init__()
        self.phi_shape = phi_shape
        p, T = phi_shape

        # 1D CNN to process each of the p rows temporally
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Pool across the temporal dimension
        )

        # Encoder to create a single characteristic value per row
        self.encoder = nn.Linear(4, 1)

        # MLP for the final prediction
        # Input size: p (from encoded Phi) + v + T_stop + L
        self.mlp = nn.Sequential(
            nn.Linear(p + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, Phi, v, T_stop, L):
        """
        Forward pass through the network.

        Parameters
        ----------
        Phi : torch.Tensor
            Phi matrix of shape (batch_size, p, T)
        v : torch.Tensor
            Threshold value of shape (batch_size,)
        T_stop : torch.Tensor
            T_stop value of shape (batch_size,)
        L : torch.Tensor
            Number of dummies of shape (batch_size,)

        Returns
        -------
        torch.Tensor
            Predicted FDP value
        """
        batch_size, p, T = Phi.shape

        # Reshape to (batch_size * p, 1, T) to apply 1D CNN to each row
        phi_reshaped = Phi.view(batch_size * p, 1, T)

        # Process each row with the CNN
        cnn_out = self.cnn(phi_reshaped)  # -> (batch_size * p, 16, 1)

        # Flatten the CNN output for the encoder
        cnn_out_flat = cnn_out.view(batch_size * p, -1)  # -> (batch_size * p, 16)

        # Encode each row to a single value
        encoded_rows = self.encoder(cnn_out_flat)  # -> (batch_size * p, 1)

        # Reshape to get the p-dimensional characteristic vector for each batch item
        char_vector = encoded_rows.view(batch_size, p)  # -> (batch_size, p)

        # Prepare v, T_stop, and L for concatenation
        v = v.unsqueeze(1)
        T_stop = T_stop.unsqueeze(1)
        L = L.unsqueeze(1)

        # Concatenate the characteristic vector and other inputs
        mlp_input = torch.cat((char_vector, v, T_stop, L), dim=1)

        # Pass through the final MLP
        output = self.mlp(mlp_input)

        return output


class AsymmetricMSELoss(nn.Module):
    def __init__(self, underestimation_weight=2.0):
        super().__init__()
        assert underestimation_weight > 1.0, "underestimation_weight must be > 1.0"
        self.w = underestimation_weight
    
    def forward(self, outputs, targets):
        overestimation = torch.relu(outputs - targets) ** 2
        underestimation = torch.relu(targets - outputs) ** 2
        return torch.mean(overestimation + self.w * underestimation)


def save_model(model, path='model.pth'):
    """
    Save the model state dictionary to a file.
    
    Parameters
    ----------
    model : nn.Module
        The model to save
    path : str, default='model.pth'
        Path where the model will be saved
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(input_size, path='model.pth', device=None):
    """
    Load a model from a saved state dictionary file.
    
    Parameters
    ----------
    input_size : tuple
        Shape of the Phi matrix (rows, columns) for initializing the model
    path : str, default='model.pth'
        Path to the saved model file
    device : torch.device, default=None
        Device to load the model to. If None, uses CUDA if available
        
    Returns
    -------
    nn.Module
        The loaded model in evaluation mode
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = FDPNet(input_size).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")
    return model
