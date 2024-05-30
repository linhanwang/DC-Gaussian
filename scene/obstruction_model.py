import tinycudann as tcnn
import torch
from torch import nn


class NOM(nn.Module):
    """
    Naive obstruction modeling
    """
    def __init__(self,
                 num_layers: int = 2,
                 hidden_dim: int = 64,
                 num_levels: int = 16,
                 features_per_level: int = 2,
                 log2_hashmap_size: int = 16,
                 base_resolution: int = 16,
                 feature_dim: int = 64,
                 network_activation: str = 'ReLU',
                 feature_output_activation: str = 'Tanh') -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.feature_output_activation = feature_output_activation

        self.encoding = tcnn.Encoding(n_input_dims=3,
                                      encoding_config={
                                          'otype': 'SequentialGrid',
                                          'n_levels': num_levels,
                                          'n_features_per_level': features_per_level,
                                          'log2_hashmap_size': log2_hashmap_size,
                                          'base_resolution': base_resolution,
                                          'include_static': False
                                      })

        self.mlp_head = tcnn.Network(
            n_input_dims=features_per_level * num_levels,
            n_output_dims=3,
            network_config={
                'otype': 'FullyFusedMLP',
                'activation': network_activation,
                'output_activation': 'Sigmoid',
                'n_neurons': hidden_dim,
                'n_hidden_layers': num_layers - 1,
            },
        )

    def forward(self, coords: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        embedding = self.encoding(coords)

        return self.mlp_head(embedding)


class IOM(nn.Module):
    """
    Illumination-aware obstruction modeling
    """
    
    def __init__(self,
                 num_layers: int = 2,
                 hidden_dim: int = 64,
                 num_levels: int = 16,
                 features_per_level: int = 2,
                 log2_hashmap_size: int = 16,
                 base_resolution: int = 16,
                 network_activation: str = 'ReLU') -> None:
        super().__init__()

        self.encoding = tcnn.Encoding(n_input_dims=3,
                                      encoding_config={
                                          'otype': 'SequentialGrid',
                                          'n_levels': num_levels,
                                          'n_features_per_level': features_per_level,
                                          'log2_hashmap_size': log2_hashmap_size,
                                          'base_resolution': base_resolution,
                                          'include_static': False
                                      })

        self.omega_head = tcnn.Network(
            n_input_dims=features_per_level * num_levels + 3,
            n_output_dims=features_per_level * num_levels,
            network_config={
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': 'Sigmoid',
                'n_neurons': hidden_dim,
                'n_hidden_layers': num_layers - 1,
            },
        )

        self.beta_head = tcnn.Network(
            n_input_dims=features_per_level * num_levels + 3,
            n_output_dims=features_per_level * num_levels,
            network_config={
                'otype': 'FullyFusedMLP',
                'activation': 'ReLU',
                'output_activation': 'Sigmoid',
                'n_neurons': hidden_dim,
                'n_hidden_layers': num_layers - 1,
            },
        )

        self.mlp_head = tcnn.Network(
            n_input_dims=features_per_level * num_levels,
            n_output_dims=3,
            network_config={
                'otype': 'FullyFusedMLP',
                'activation': network_activation,
                'output_activation': 'Sigmoid',
                'n_neurons': hidden_dim,
                'n_hidden_layers': num_layers - 1,
            },
        )

    def forward(self, coords: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        embedding = self.encoding(coords)
        t = T.unsqueeze(0).repeat(embedding.size()[0], 1)
        cat_feat = torch.cat([embedding, t], -1)

        gamma = self.omega_head(cat_feat)
        beta = self.beta_head(cat_feat)
        return self.mlp_head(gamma * embedding + beta)


obs_dict = {
    'nom': NOM,
    'iom': IOM,
}
