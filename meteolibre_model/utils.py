
import torch.nn as nn

class FilmLayer(nn.Module):
    """
    FILM layer: Feature-wise Linear Modulation.
    Applies scale and bias to the input features based on the conditioning input.
    """
    def __init__(self, num_features, condition_size):
        super().__init__()
        self.num_features = num_features
        # Linear layers to predict scale and bias from the condition vector
        self.film_dense = nn.Linear(condition_size, 2 * num_features)

    def forward(self, x, condition):
        """
        Args:
            x (torch.Tensor): Input feature map (B, C, H, W).
            condition (torch.Tensor): Conditioning vector (B, condition_size).

        Returns:
            torch.Tensor: Modulated feature map.
        """
        film_params = self.film_dense(condition)
        gamma, beta = film_params[:, :self.num_features], film_params[:, self.num_features:]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1) # (B, C, 1, 1) to match feature map dims
        beta = beta.unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)

        return gamma * x + beta
