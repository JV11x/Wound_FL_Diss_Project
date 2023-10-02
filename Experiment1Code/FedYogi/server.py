from typing import List, Tuple
import flwr as fl
from flwr.common import Metrics, FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from model import UNET
import numpy as np
import torch
from typing import List, Tuple, Union, Optional, Dict
import glob
import os
from collections import OrderedDict


#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the UNET model
model = UNET(in_channels=3, out_channels=1)

def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

params = get_parameters(model)

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    dice_scores = [num_examples * m["dice_score"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)

    return {"dice_score_avg": sum(dice_scores) / sum(examples)}

class SaveModelStrategy(fl.server.strategy.FedYogi):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

            # Save the model
            torch.save(model.state_dict(), f"model_round_{server_round}.pth")

        return aggregated_parameters, aggregated_metrics

# Define strategy with initial_parameters
strategy = SaveModelStrategy(
    initial_parameters=fl.common.ndarrays_to_parameters(params),  # Include initial parameters here
    evaluate_metrics_aggregation_fn=weighted_average,
    )

if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080",  
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )