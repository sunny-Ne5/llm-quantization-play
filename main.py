import argparse
from llm_quantization import Model, ModelManager, MetricsTracker


def main():

    parser = argparse.ArgumentParser(
        description="Run a GPT-2 model with optional quantization in a REPL."
    )
    parser.add_argument(
        "model_name", type=str, help="Name of the GPT-2 model (e.g., gpt2)"
    )
    parser.add_argument(
        "quantization",
        type=str,
        choices=["none", "8bit", "4bit", "16bit"],
        help="Quantization type",
    )
    args = parser.parse_args()

    model = Model(
        args.model_name, args.quantization if args.quantization != "none" else None
    )
    model_manager = ModelManager(model)
    model_manager.apply_quantization(model.quantization_strategy)
    metrics_tracker = MetricsTracker(model_manager)

    print("Starting REPL. Type 'exit' to quit.")
    while True:
        try:
            input_text = input("Enter text: ")
            if input_text.lower() == "exit":
                break
            metrics = metrics_tracker.track_inference(input_text)
            print(f"Output: {metrics['output']}")
            print(f"Latency: {metrics['latency_seconds']:.2f} sec")
            for gpu_id, memory_used in metrics["memory_used_mb"].items():
                print(f"GPU {gpu_id} memory used: {memory_used:.2f} MB")
        except KeyboardInterrupt:
            break
    print("Exiting REPL.")


if __name__ == "__main__":
    main()
