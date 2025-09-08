# Import objects from KubeFlow Pipelines DSL Library
from kfp.dsl import (
    component,
    Input,
    Output,
    Model,
    Artifact,
)


@component(base_image="pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime",
           packages_to_install=["pandas==2.2.3",
                                "dask[dataframe]==2024.8.0",
                                "scikit-learn==1.6.1",
                                "onnxscript"])
def convert_torch_to_onnx(
    version: str,
    torch_model: Input[Model],
    hyperparameters: Input[Artifact],
    onnx_model: Output[Model],
):
    # export model to onnx
    from torch import randn
    import torch.cuda as tc
    import torch.onnx
    # Import torch specific libraries
    from torch.nn import Module, Sequential, ReLU, Linear
    from collections import OrderedDict

    """ Define the Neural Network Architecture """

    # build the neural network that will predict the country out of selected features
    class CountryPredictorNetwork(Module):
        def __init__(self, n_inputs, hidden_len, n_outputs):
            super().__init__()
            self.predictor_network = Sequential(OrderedDict([
                ("input", Linear(n_inputs, hidden_len)),
                ("input_activation", ReLU()),
                ("linear_0", Linear(hidden_len, hidden_len * 2)),
                ("activation_0", ReLU()),
                ("linear_1", Linear(hidden_len * 2, hidden_len * 4)),
                ("activation_1", ReLU()),
                ("linear_2", Linear(hidden_len * 4, hidden_len * 8)),
                ("activation_2", ReLU()),
                ("output", Linear(hidden_len * 8, n_outputs)),
            ]))

        def forward(self, input_sample):
            return self.predictor_network(input_sample)

    # load parameters
    import json
    parms = {}
    with open(hyperparameters.path, "r") as p:
        parms = json.load(p)

    # get parms
    n_feats = parms.get("n_feats")
    base_size = parms.get("base_size")
    n_classes = parms.get("n_classes")

    # instantiate a model object
    model = CountryPredictorNetwork(n_feats, base_size, n_classes)
    # load pretrained weights from checkpoint
    model.load_state_dict(torch.load(torch_model.path))

    # detect device
    device = "cpu"
    if tc.is_available():
        device = "cuda"

    print(f"DEVICE:\n Converting on {device}")

    # Convert & save model
    input_signature = randn(1, n_feats).to(device)
    onnx_converted_model = torch.onnx.export(model, input_signature, dynamo=True)

    # save ONNX checkpoint
    onnx_model._set_path(onnx_model.path + ".onnx")
    onnx_converted_model.save(onnx_model.path)