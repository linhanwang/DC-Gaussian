import tinycudann as tcnn
import torch
from torch import nn


class SkyField(nn.Module):

    def __init__(
            self,
            num_layers: int = 2,
            hidden_dim: int = 64,
            num_levels: int = 16,
            features_per_level: int = 2,
            log2_hashmap_size: int = 16,
            base_resolution: int = 16,
            feature_dim: int = 64,
            network_activation: str = 'ReLU',
            feature_output_activation: str = 'Tanh'
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.feature_output_activation = feature_output_activation

        self.encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                'otype': 'SequentialGrid',
                'n_levels': num_levels,
                'n_features_per_level': features_per_level,
                'log2_hashmap_size': log2_hashmap_size,
                'base_resolution': base_resolution,
                'include_static': False
            }
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

    def forward(self, dirs: torch.Tensor) -> torch.Tensor:
        embedding = self.encoding(dirs)

        return self.mlp_head(embedding)
