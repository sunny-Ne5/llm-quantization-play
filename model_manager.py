from .quantization.quantization_engine import QuantizationEngine

class ModelManager:
    def __init__(self, model):
        """Initialize the ModelManager.

        Args:
            model (Model): The model to manage.
        """
        self.model = model
        self.quantization_engine = None

    def apply_quantization(self, qStrategy):
        """Apply quantization to the model.

        Args:
            qStrategy: Quantization strategy (e.g., 8-bit, 4-bit).
        """
        print(qStrategy)
        self.quantization_engine = QuantizationEngine(self.model.model, qStrategy)
        self.quantization_engine.quantize()

    def infer(self, input_text: str, max_length: int = 50) -> str:
        """Run inference on the model.

        Args:
            input_text (str): Input text for inference.
            max_length (int): Maximum length of the generated text.

        Returns:
            str: Generated text.
        """
        return self.model.generate(input_text, max_length)
