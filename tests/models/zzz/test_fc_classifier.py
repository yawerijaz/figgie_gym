import torch

from figgie_gym.models.zzz.fc_classifier import FCClassifier


def test_forward_shape():
    input_dim = 16
    num_classes = 4
    model = FCClassifier(
        input_dim=input_dim, hidden_dims=[32], num_classes=num_classes
    )
    x = torch.randn(2, input_dim)
    out = model(x)
    assert out.shape == (2, num_classes)
