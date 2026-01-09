import torch
from src.config import ProjectConfig
from src.modeling.architecture import InterpretableMLP
from src.features import interpreter_factory
from src.plots import VisualizationEngine
from src.dataset import MNISTDataModule


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
        self, 
        model_filename: str, 
        input_image: torch.Tensor, 
        target_label: int,
        interpreter_type: str  # <--- ZMIANA: Przekazujemy typ jawnie
    ) -> None:
        print(f"Analyzing: {model_filename} using {interpreter_type}...")
        model = self.load_model(model_filename)
        
        # Tworzymy interpreter dynamicznie na podstawie argumentu
        interpreter = interpreter_factory(interpreter_type, model)

        # 1. Structural Analysis (Visualize Weights)
        # Wagi są niezależne od interpretera, więc robimy to raz (lub nadpisujemy)
        fc1_weights = interpreter.get_layer_weights("fc1")
        self.visualizer.plot_weight_matrix(
            fc1_weights, f"Weights ({model_filename})", f"weights_{model_filename}.png"
        )

        # 2. Decision Process (Saliency Map)
        saliency = interpreter.compute_saliency_map(input_image, target_label)
        
        # <--- ZMIANA: Nazwa pliku zawiera teraz typ interpretera!
        # Np. saliency_standard_mlp.pth_captum_ig.png
        safe_name = interpreter_type.replace("_", "-")
        filename = f"saliency_{model_filename}_{safe_name}.png"
        
        self.visualizer.plot_saliency(
            input_image, saliency, filename
        )
        print(f"Saved visualization to {filename}")


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

    # <--- ZMIANA: Uruchamiamy analizę dla KAŻDEGO interpretera po kolei
    # Dzięki temu wygenerujemy komplet obrazków do porównania
    interpreters_to_test = ["integrated_grad", "captum_ig", "captum_deeplift"]
    
    models_to_test = ["standard_mlp.pth", "sparse_mlp.pth"]

    for model_name in models_to_test:
        for interp in interpreters_to_test:
            try:
                runner.analyze_model(model_name, sample_img, sample_label, interpreter_type=interp)
            except FileNotFoundError:
                print(f"Model {model_name} not found. Skip.")
            except Exception as e:
                print(f"Error analyzing {model_name} with {interp}: {e}")


if __name__ == "__main__":
    run_analysis()