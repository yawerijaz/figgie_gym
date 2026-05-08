import torch

from figgie_gym.agent.supervised import (
    get_device,
)


class TestGetDevice:
    """Test the get_device utility function."""

    def test_get_device_returns_device(self) -> None:
        """Test that get_device returns a torch.device."""
        device = get_device()
        assert isinstance(device, torch.device)  # pyright: ignore[reportPrivateImportUsage]

    def test_get_device_cpu_available(self) -> None:
        """Test that CPU device is always available."""
        device = get_device()
        # Should return some valid device
        assert device.type in ["cuda", "mps", "cpu"]

    def test_get_device_consistent(self) -> None:
        """Test that get_device returns consistent device."""
        device1 = get_device()
        device2 = get_device()
        assert device1.type == device2.type
