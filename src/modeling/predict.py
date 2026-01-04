import torch
from config import ProjectConfig
from modeling.architecture import InterpretableMLP
from features import interpreter_factory
from plots import VisualizationEngine
from dataset import MNISTDataModule


class InferenceRunner:
    """
    Orchestrates the prediction and interpretation pipeline.
    """

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.visualizer = VisualizationEngine(config.FIGURES_DIR)

    def load_model(self, filename: str) -> InterpretableMLP:
        model = InterpretableMLP(
            self.config.INPUT_SIZE, self.config.HIDDEN_SIZE, self.config.OUTPUT_SIZE
        )
        path = self.config.MODELS_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"Model {filename} not found. Run train.py first.")

        model.load_state_dict(torch.load(path))
        return model

    def analyze_model(
        self, model_filename: str, input_image: torch.Tensor, target_label: int
    ) -> None:
        print(f"Analyzing: {model_filename}...")
        model = self.load_model(model_filename)
        interpreter = interpreter_factory(self.config.INTERPRETER, model)

        # 1. Structural Analysis (Visualize Weights)
        # This checks if regularization successfully sparsified the connections
        fc1_weights = interpreter.get_layer_weights("fc1")
        self.visualizer.plot_weight_matrix(
            fc1_weights, f"Weights ({model_filename})", f"weights_{model_filename}.png"
        )

        # 2. Decision Process (Saliency Map)
        # This checks which pixels the model focused on
        saliency = interpreter.compute_saliency_map(input_image, target_label)
        self.visualizer.plot_saliency(
            input_image, saliency, f"saliency_{model_filename}.png"
        )


def run_analysis() -> None:
    config = ProjectConfig()
    runner = InferenceRunner(config)

    # Get a sample image
    data_module = MNISTDataModule(config)
    _, test_loader = data_module.get_data_loaders()
    images, labels = next(iter(test_loader))

    # Select first image
    sample_img = images[0:1]
    sample_label = labels[0].item()

    # Analyze Standard Model
    try:
        runner.analyze_model("standard_mlp.pth", sample_img, sample_label)
    except FileNotFoundError:
        print("Standard model not found.")

    # Analyze Sparse Model
    try:
        runner.analyze_model("sparse_mlp.pth", sample_img, sample_label)
    except FileNotFoundError:
        print("Sparse model not found.")


if __name__ == "__main__":
    run_analysis()
