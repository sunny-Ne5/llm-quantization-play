import torch
import pynvml
from typing import Dict
import time


class MetricsTracker:
    def __init__(self, model_manager):
        """Initialize the MetricsTracker.

        Args:
            model_manager (ModelManager): The ModelManager to track.
        """
        self.model_manager = model_manager
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialize_nvml()

    def initialize_nvml(self):
        """Initialize the NVIDIA Management Library (NVML)."""
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as e:
            print(f"NVML initialization failed: {str(e)}")

    def shutdown_nvml(self):
        """Shutdown the NVIDIA Management Library (NVML)."""
        pynvml.nvmlShutdown()

    def get_gpu_memory_usage(self) -> Dict[int, Dict[str, float]]:
        """Get GPU memory usage for all available GPUs.

        Returns:
            Dict[int, Dict[str, float]]: Dictionary containing memory usage for each GPU.
        """
        self.initialize_nvml()
        gpu_memory = {}
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory[i] = {
                "total_memory_mb": info.total / (1024 * 1024),
                "used_memory_mb": info.used / (1024 * 1024),
                "free_memory_mb": info.free / (1024 * 1024),
            }
        return gpu_memory

    def get_gpu_utilization(self) -> Dict[int, Dict[str, float]]:
        """Get GPU utilization for all available GPUs.

        Returns:
            Dict[int, Dict[str, float]]: Dictionary containing utilization for each GPU.
        """
        self.initialize_nvml()
        gpu_utilization = {}
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization[i] = {
                "gpu_utilization_percent": util.gpu,
                "memory_utilization_percent": util.memory,
            }
        return gpu_utilization

    def track_inference(
        self, input_text: str, max_length: int = 50
    ) -> Dict[str, float]:
        """Track GPU memory usage, utilization, and latency during inference.

        Args:
            input_text (str): Input text for inference.
            max_length (int): Maximum length of the generated text.

        Returns:
            Dict[str, float]: Dictionary containing GPU metrics and latency.
        """
        self.initialize_nvml()
        # Track GPU memory before inference
        gpu_memory_before = self.get_gpu_memory_usage()
        gpu_utilization_before = self.get_gpu_utilization()

        # Track latency
        start_time = time.time()
        output = self.model_manager.infer(input_text, max_length)
        latency = time.time() - start_time

        # Track GPU memory after inference
        gpu_memory_after = self.get_gpu_memory_usage()
        gpu_utilization_after = self.get_gpu_utilization()

        # Calculate memory used during inference
        memory_used_mb = {}
        for gpu_id in gpu_memory_before:
            memory_used_mb[gpu_id] = (
                gpu_memory_after[gpu_id]["used_memory_mb"]
                - gpu_memory_before[gpu_id]["used_memory_mb"]
            )

        # Shutdown NVML
        self.shutdown_nvml()

        return {
            "memory_used_mb": memory_used_mb,
            "gpu_utilization_before": gpu_utilization_before,
            "gpu_utilization_after": gpu_utilization_after,
            "latency_seconds": latency,
            "output": output,
        }
