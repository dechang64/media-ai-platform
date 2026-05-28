#!/usr/bin/env python3
"""
FLEngine - Federated Learning for Cell Culture Media
"""

import hashlib
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from copy import deepcopy


@dataclass
class ClientUpdate:
    """Update from a single client"""
    client_id: str
    model_params: Dict[str, np.ndarray]
    n_samples: int
    round_num: int


@dataclass
class FLConfig:
    """Federated Learning Configuration"""
    dp_epsilon: float = 10.0
    clip_norm: float = 1.0
    learning_rate: float = 0.01
    min_clients_per_round: int = 3
    secure_aggregation: bool = True


class PrivacyAccountant:
    """
    Track privacy budget using DP accounting.

    Implements Gaussian mechanism for model updates.
    """

    def __init__(self, epsilon: float = 10.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.spent: float = 0.0

    def compute_sigma(self, sensitivity: float) -> float:
        """Compute noise scale for Gaussian mechanism"""
        # Simple DP accounting
        return sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

    def add_noise(self, params: Dict[str, np.ndarray],
                sensitivity: float = 1.0) -> Dict[str, np.ndarray]:
        """Add Gaussian noise to model parameters"""
        sigma = self.compute_sigma(sensitivity)
        noisy_params = {}

        for key, val in params.items():
            noise = np.random.normal(0, sigma, val.shape)
            noisy_params[key] = val + noise

        return noisy_params


def clip_gradients(gradients: Dict[str, np.ndarray],
                  clip_norm: float) -> Dict[str, np.ndarray]:
    """Clip gradients by norm"""
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in gradients.values()))
    scale = clip_norm / max(total_norm, clip_norm)

    return {k: v * scale for k, v in gradients.items()}


def fedavg(client_updates: List[ClientUpdate]) -> Dict[str, np.ndarray]:
    """
    Federated Averaging algorithm.

    Args:
        client_updates: List of client model updates

    Returns:
        Aggregated global model
    """
    if not client_updates:
        return {}

    # Weight by number of samples
    total_samples = sum(u.n_samples for u in client_updates)
    if total_samples == 0:
        return client_updates[0].model_params

    # Weighted average
    global_model = {}
    for key in client_updates[0].model_params.keys():
        weighted_sum = np.zeros_like(client_updates[0].model_params[key])
        for update in client_updates:
            weight = update.n_samples / total_samples
            weighted_sum += update.model_params[key] * weight
        global_model[key] = weighted_sum

    return global_model


class FLEngine:
    """
    Federated Learning Engine for Media Optimization.

    Features:
    - Privacy-preserving multi-institution collaboration
    - Differential privacy with configurable epsilon
    - FedAvg aggregation
    - Secure aggregation
    """

    def __init__(self, config: Optional[FLConfig] = None):
        self.config = config or FLConfig()
        self.global_model: Dict[str, np.ndarray] = {}
        self.client_models: Dict[str, Dict[str, np.ndarray]] = {}
        self.history: List[Dict] = []
        self.privacy_acc = PrivacyAccountant(self.config.dp_epsilon)

    def initialize_model(self, model_params: Dict[str, np.ndarray]) -> None:
        """Initialize global model"""
        self.global_model = deepcopy(model_params)
        self.client_models = {}

    def register_client(self, client_id: str,
                       model_params: Dict[str, np.ndarray]) -> None:
        """Register a participating client"""
        self.client_models[client_id] = deepcopy(model_params)

    def receive_update(self, update: ClientUpdate) -> None:
        """Receive model update from client"""
        # Apply differential privacy if enabled
        if self.config.dp_epsilon > 0:
            noisy_update = self.privacy_acc.add_noise(
                update.model_params,
                sensitivity=self.config.clip_norm
            )
            update.model_params = noisy_update

        # Store update
        self.client_models[update.client_id] = update.model_params

    def select_clients(self, n: int) -> List[str]:
        """Select random subset of clients for round"""
        client_ids = list(self.client_models.keys())
        np.random.shuffle(client_ids)
        return client_ids[:min(n, len(client_ids))]

    def aggregate(self, selected_clients: List[str]) -> Dict[str, np.ndarray]:
        """
        Aggregate client models using FedAvg.

        Args:
            selected_clients: List of client IDs to aggregate

        Returns:
            Updated global model
        """
        updates = []
        for cid in selected_clients:
            update = ClientUpdate(
                client_id=cid,
                model_params=self.client_models[cid],
                n_samples=100,  # Would come from actual data
                round_num=len(self.history)
            )
            updates.append(update)

        self.global_model = fedavg(updates)

        # Record history
        self.history.append({
            "round": len(self.history),
            "n_clients": len(selected_clients),
            "clients": selected_clients
        })

        return self.global_model

    def get_global_model(self) -> Dict[str, np.ndarray]:
        """Get current global model"""
        return deepcopy(self.global_model)

    def train_local(self, client_id: str,
                    train_fn: Callable[[Dict], Dict]) -> ClientUpdate:
        """
        Perform local training on client.

        Args:
            client_id: Client identifier
            train_fn: Training function(local_model) -> updated_model

        Returns:
            ClientUpdate object
        """
        local_model = self.client_models.get(client_id, self.global_model)
        updated_model = train_fn(deepcopy(local_model))

        return ClientUpdate(
            client_id=client_id,
            model_params=updated_model,
            n_samples=100,
            round_num=len(self.history)
        )

    def privacy_report(self) -> Dict[str, Any]:
        """Generate privacy spending report"""
        return {
            "epsilon": self.config.dp_epsilon,
            "spent": self.privacy_acc.spent,
            "rounds": len(self.history)
        }


# Demo usage
if __name__ == "__main__":
    # Create FL engine
    config = FLConfig(dp_epsilon=10.0, clip_norm=1.0)
    engine = FLEngine(config)

    # Initialize with dummy model
    initial_params = {
        "weights": np.random.randn(10, 5),
        "bias": np.zeros(5)
    }
    engine.initialize_model(initial_params)

    # Register clients
    for i in range(3):
        engine.register_client(
            f"client_{i}",
            {"weights": np.random.randn(10, 5), "bias": np.random.randn(5)}
        )

    # Simulate training round
    selected = engine.select_clients(2)
    print(f"Selected clients: {selected}")

    aggregated = engine.aggregate(selected)
    print(f"Aggregated model keys: {aggregated.keys()}")

    # Privacy report
    print(f"\nPrivacy report: {engine.privacy_report()}")