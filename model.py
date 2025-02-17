import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from .quantization.quantization_strategy import (
    Simple8BitQuantizationStrategy,
    Simple4BitQuantizationStrategy,
    Simple16BitQuantizationStrategy,
)

class Model:
    def __init__(self, model_name: str, quantization: str):
        self.model_name = model_name
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.quantization_strategy = self.get_quantization_strategy(quantization)
        # if self.quantization_strategy:
        #     self.apply_quantization()

    def get_quantization_strategy(self, quantization: str):
        if quantization == "8bit":
            return Simple8BitQuantizationStrategy()
        elif quantization == "4bit":
            return Simple4BitQuantizationStrategy()
        elif quantization == "16bit":
            return Simple16BitQuantizationStrategy()
        return None

    def apply_quantization(self):
        for name, param in self.model.named_parameters():
            if param.dim() >= 2:
                quantized_data = self.quantization_strategy.quantize(param.data)
                param.data = self.quantization_strategy.dequantize(
                    quantized_data["quantized_tensor"],
                    quantized_data["scale"],
                    quantized_data["zero_point"],
                )

    def generate(self, input_text: str, max_length: int = 50) -> str:
        inputs = self.tokenizer(
            input_text, return_tensors="pt", padding=True, truncation=True
        )
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
