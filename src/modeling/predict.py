import torch
from src.config import ProjectConfig
from src.modeling.architecture import InterpretableMLP, SimpleCNN
from src.features import interpreter_factory, INTERPRETER_REGISTRY
from src.plots import VisualizationEngine
from src.dataset import MNISTDataModule


class InferenceRunner:
    """
    Orchestrates the prediction and interpretation pipeline.
    Handles both MLP (vector input) and CNN (image input).
    """

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.visualizer = VisualizationEngine(config.FIGURES_DIR)

    def load_model(self, filename: str) -> torch.nn.Module:
        if "cnn" in filename:
            model = SimpleCNN()
        else:
            model = InterpretableMLP(
                self.config.INPUT_SIZE, self.config.HIDDEN_SIZE, self.config.OUTPUT_SIZE
            )

        path = self.config.MODELS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Model {filename} not found. Run train.py first.")

        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        return model

    def analyze_model(
        self, 
        model_filename: str, 
        input_image: torch.Tensor, 
        target_label: int,
        interpreter_type: str
    ) -> None:
        print(f"Analyzing: {model_filename} using {interpreter_type}...")
        
        model = self.load_model(model_filename)
        
        if isinstance(model, SimpleCNN):
            input_tensor = input_image
        else:
            input_tensor = input_image.view(1, -1)

        interpreter = interpreter_factory(interpreter_type, model)
 
        if not isinstance(model, SimpleCNN):
            try:
                fc1_weights = interpreter.get_layer_weights("fc1")
                self.visualizer.plot_weight_matrix(
                    fc1_weights, 
                    f"Weights ({model_filename})", 
                    f"weights_{model_filename}.png"
                )
            except Exception as e:
                pass

        safe_method_name = interpreter_type.replace("_", "-")
        filename = f"saliency_{model_filename}_{safe_method_name}.png"

        try:
            saliency = interpreter.compute_saliency_map(input_tensor, target_label)
            
            if saliency.dim() == 2 and saliency.shape[1] == 784:
                saliency = saliency.view(1, 1, 28, 28)

            self.visualizer.plot_saliency(
                input_image, saliency, filename
            )
            print(f" -> Saved: {filename}")
        except Exception as e:
            print(f" -> Error computing saliency with {interpreter_type}: {e}")


def run_analysis() -> None:
    config = ProjectConfig()
    runner = InferenceRunner(config)

    data_module = MNISTDataModule(config)
    _, test_loader = data_module.get_data_loaders()
    
    images, labels = next(iter(test_loader))
    sample_img = images[0:1]
    sample_label = labels[0].item()

    print(f"Selected Sample Label: {sample_label}")

    models_to_test = ["standard_mlp.pth", "sparse_mlp.pth", "simple_cnn.pth"]
    interpreters_to_test = ["integrated_grad", "captum_ig", "captum_saliency"]

    for model_name in models_to_test:
        print(f"\n--- Processing Model: {model_name} ---")
        for interp in interpreters_to_test:
            try:
                runner.analyze_model(
                    model_filename=model_name, 
                    input_image=sample_img, 
                    target_label=sample_label,
                    interpreter_type=interp
                )
            except FileNotFoundError:
                print(f"Model file {model_name} missing. Skipping.")
            except Exception as e:
                print(f"Critical error analyzing {model_name}: {e}")


if __name__ == "__main__":
    run_analysis()