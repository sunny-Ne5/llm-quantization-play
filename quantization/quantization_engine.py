import torch
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

class QuantizationEngine:
    """Responsibility: Quantization handling
    """
    def __init__(self, model, qStrategy):
        """Init method

        Args:
            model (torch.nn.Module): The model to quantize.
            qStrategy: Quantization strategy (e.g., 8-bit, 4-bit).
        """
        self.model = model
        self.tensors: Dict[str, torch.Tensor] = {}
        self.quantized_tensors: Dict[str, Dict[str, torch.Tensor]] = {}
        self.qStrategy = qStrategy

    def quantize(self):
        """Quantize the model permanently and replace original weights."""
        print(f"Starting {self.qStrategy.bit_width}bit quantization")
        self.__extract_tensors_for_quantization()

        original_size = sum(p.numel() * p.element_size() for p in self.model.parameters())

        # Quantize tensors in parallel
        self.quantized_tensors = self.__parallel_quantize_tensors()

        # Replace model parameters with quantized versions
        for name, module in self.model.named_modules():
            if name in self.quantized_tensors:
                quantized_data = self.quantized_tensors[name]

                # Free original FP32 weights to save space
                del module.weight

                # Convert quantized data back to a format PyTorch can use
                module.register_parameter(
                    "weight",
                    torch.nn.Parameter(
                        self.qStrategy.dequantize(
                            quantized_data["quantized_tensor"],
                            quantized_data["scale"],
                            quantized_data["zero_point"],
                        ),
                        requires_grad=False,
                    ),
                )
                print(f"Replaced {name} with quantized weights.")

        quantized_size = sum(
            tensor["quantized_tensor"].numel() for tensor in self.quantized_tensors.values()
        ) * (self.qStrategy.bit_width / 8)  # Quantized size

        print(f"Original model size: {original_size / (1024 * 1024):.2f} MB")
        print(f"Quantized model size: {quantized_size / (1024 * 1024):.2f} MB")
        print(f"Size reduction: {(original_size - quantized_size) / (1024 * 1024):.2f} MB")


    @staticmethod
    def __should_quantize_tensor(tensor_name: str, tensor_shape: torch.Size) -> bool:
        """
        Determine if a tensor should be quantized based on its name, shape, and parameters.

        Args:
            tensor_name (str): Name of the tensor.
            tensor_shape (torch.Size): Shape of the tensor.

        Returns:
            bool: True if the tensor should be quantized, False otherwise.
        """
        # Do not quantize embeddings or LayerNorm layers
        if "wte" in tensor_name or "wpe" in tensor_name:
            return False  # Keep word and positional embeddings in full precision
        if "ln_" in tensor_name:
            return False  # Keep LayerNorm layers in full precision

        # Quantize only linear layers with 2D weight matrices
        return len(tensor_shape) == 2


    def __extract_tensors_for_quantization(self):
        """Extract tensors from the model that should be quantized."""
        for name, module in self.model.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                tensor_name = name
                tensor_shape = module.weight.shape
                if self.__should_quantize_tensor(tensor_name, tensor_shape):
                    self.tensors[tensor_name] = module.weight.data

    def __quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Quantize a single tensor using the quantization strategy.

        Args:
            tensor (torch.Tensor): The tensor to quantize.

        Returns:
            torch.Tensor: Quantized tensor.
        """
        return self.qStrategy.quantize(tensor)

    def __parallel_quantize_tensors(self) -> Dict[str, torch.Tensor]:
        """
        Quantize tensors in parallel using ThreadPoolExecutor.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of quantized tensors.
        """
        quantized_tensors = {}
        with ThreadPoolExecutor() as executor:
            futures = {name: executor.submit(self.__quantize_tensor, tensor) for name, tensor in self.tensors.items()}
            for name, future in futures.items():
                quantized_tensors[name] = future.result()
        return quantized_tensors
