import torch
from typing import Dict

class BaseQuantizationStrategy:
    def __init__(self, bit_width: int):
        """
        Base class for quantization strategies.

        Args:
            bit_width (int): Bit width for quantization (e.g., 4, 8, 16).
        """
        self.bit_width = bit_width

    def quantize(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement the quantize method.")

    def dequantize(self, quantized_tensor: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
        """
        Dequantize a tensor back to float32.
        """
        return (quantized_tensor.float() - zero_point.float()) / scale.float()

class Simple8BitQuantizationStrategy(BaseQuantizationStrategy):
    def __init__(self):
        super().__init__(bit_width=8)

    def quantize(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        min_val = tensor.min()
        max_val = tensor.max()
        scale = 255.0 / (max_val - min_val)
        zero_point = -min_val * scale
        quantized_tensor = torch.round(tensor * scale + zero_point).to(torch.uint8)
        return {"quantized_tensor": quantized_tensor, "scale": scale.clone().detach(), "zero_point": zero_point.clone().detach()}

class Simple4BitQuantizationStrategy(BaseQuantizationStrategy):
    def __init__(self):
        super().__init__(bit_width=4)

    def quantize(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        min_val = tensor.min()
        max_val = tensor.max()
        scale = 15.0 / (max_val - min_val)  # 4-bit range: [0, 15]
        zero_point = -min_val * scale
        quantized_tensor = torch.round(tensor * scale + zero_point).to(torch.uint8)  # Stored in uint8 for simplicity
        return {"quantized_tensor": quantized_tensor, "scale": scale.clone().detach(), "zero_point": zero_point.clone().detach()}

class Simple16BitQuantizationStrategy(BaseQuantizationStrategy):
    def __init__(self):
        super().__init__(bit_width=16)

    def quantize(self, tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"quantized_tensor": tensor.half(), "scale": torch.tensor(1.0), "zero_point": torch.tensor(0.0)}
