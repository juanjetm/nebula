# nebula/core/models/covtype/mlp.py

import torch

from nebula.core.models.nebulamodel import NebulaModel


class CovtypeModelMLP(NebulaModel):
    def __init__(
        self,
        input_dim=54,
        num_classes=7,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
    ):
        # OJO: NebulaModel está pensado para imágenes (input_channels),
        # pero en la práctica se usa ese primer argumento como "input shape info".
        # Para tabular, pasamos input_dim en input_channels para mantener la firma.
        super().__init__(input_dim, num_classes, learning_rate, metrics, confusion_matrix, seed)

        # Mantengo el mismo patrón que tu MLP de FashionMNIST.
        self.config = {"beta1": 0.9, "beta2": 0.999, "amsgrad": True}

        self.example_input_array = torch.rand(1, input_dim)
        self.learning_rate = learning_rate
        self.criterion = torch.nn.CrossEntropyLoss()

        self.l1 = torch.nn.Linear(input_dim, 256)
        self.l2 = torch.nn.Linear(256, 128)
        self.l3 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        # En tabular, x debe ser (batch, input_dim).
        # A veces puede venir con dimensión extra (batch, 1, input_dim) por loaders.
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)

        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        x = torch.relu(x)
        x = self.l3(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def get_learning_rate(self) -> float:
        return float(self.learning_rate)

    def count_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))
