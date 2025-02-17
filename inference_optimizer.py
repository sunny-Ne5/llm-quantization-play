from typing import List


class InferenceOptimizer:
    def __init__(self, model_manager):
        """Initialize the InferenceOptimizer.

        Args:
            model_manager (ModelManager): The ModelManager to optimize.
        """
        self.model_manager = model_manager

    def optimize_inference(
        self, input_texts: List[str], max_length: int = 50
    ) -> List[str]:
        """Optimize inference using batching.

        Args:
            input_texts (List[str]): List of input texts for inference.
            max_length (int): Maximum length of the generated text.

        Returns:
            List[str]: List of generated texts.
        """
        # Batch processing
        outputs = []
        for input_text in input_texts:
            output = self.model_manager.infer(input_text, max_length)
            outputs.append(output)
        return outputs
