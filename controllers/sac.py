#!/usr/bin/env python3
"""
Dense Reward SAC Controller for controls_challenge evaluation.

This controller loads the trained dense reward SAC model and provides
the BaseController interface for controls_challenge evaluation.
"""

from . import BaseController
import os
import sys
import numpy as np
from pathlib import Path

# Add sac-drive to path
sac_drive_path = Path(__file__).parent.parent.parent / "sac-drive"
sys.path.insert(0, str(sac_drive_path))

try:
    import json


def find_experiment_by_name(sac_drive_path, experiment_name=None):
    """Find experiment by name or return the latest experiment."""
    experiments_dir = sac_drive_path / "experiments"
    if not experiments_dir.exists():
        return None, None

    if experiment_name:
        # Look for specific experiment
        exp_dir = experiments_dir / experiment_name
        if exp_dir.exists() and exp_dir.is_dir():
            config_path = exp_dir / "config.json"
            checkpoints_dir = exp_dir / "checkpoints"
            if config_path.exists() and checkpoints_dir.exists():
                checkpoints = list(checkpoints_dir.glob("checkpoint_step_*.zip"))
                if checkpoints:
                    latest_checkpoint = max(checkpoints)
                    return str(latest_checkpoint), str(config_path)
        return None, None
    else:
        # Find latest experiment with checkpoints
        valid_experiments = []
        for exp_dir in experiments_dir.iterdir():
            if exp_dir.is_dir():
                config_path = exp_dir / "config.json"
                checkpoints_dir = exp_dir / "checkpoints"
                if config_path.exists() and checkpoints_dir.exists():
                    checkpoints = list(checkpoints_dir.glob("checkpoint_step_*.zip"))
                    if checkpoints:
                        latest_checkpoint = max(checkpoints)
                        valid_experiments.append((exp_dir.stat().st_mtime, exp_dir, latest_checkpoint))

        if not valid_experiments:
            return None, None

        # Return the most recent experiment
        _, exp_dir, latest_checkpoint = max(valid_experiments)
        config_path = exp_dir / "config.json"
        return str(latest_checkpoint), str(config_path)


    from stable_baselines3 import SAC
    from config.training_config import TrainingConfig
    from src.features.state_processor import StateProcessor
    SAC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import SAC dependencies: {e}")
    SAC_AVAILABLE = False


class Controller(BaseController):
    """
    SAC controller for controls_challenge evaluation.

    Dynamically loads any trained SAC model for lateral control.
    """

    def __init__(self, model_path=None, experiment_name=None):
        """
        Initialize the SAC controller.

        Args:
            model_path: Optional direct path to model file. If None, will discover automatically.
            experiment_name: Optional experiment name to load. If None, loads latest experiment.
        """

        if not SAC_AVAILABLE:
            raise ImportError("SAC controller dependencies not available")

        # Discover model path dynamically
        if model_path is None:
            discovered_model_path, discovered_config_path = find_experiment_by_name(sac_drive_path, experiment_name)
            if discovered_model_path is None:
                if experiment_name:
                    raise FileNotFoundError(f"Experiment '{experiment_name}' not found or has no checkpoints.")
                else:
                    raise FileNotFoundError("No experiments found with checkpoints. Please train a model first.")
            model_path = discovered_model_path
            config_path = discovered_config_path
        else:
            # If model_path is provided, derive config_path
            model_dir = Path(model_path).parent.parent
            config_path = str(model_dir / "config.json")

        experiment_info = f" (experiment: {experiment_name})" if experiment_name else ""
        print(f"Loading SAC model from: {model_path}{experiment_info}")

        # Load configuration
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            self.config = TrainingConfig.from_dict(config_dict)
            print(f"Config loaded - Reward type: {self.config.reward_type}")
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
            # Use defaults
            self.config = TrainingConfig()

        # Load the trained SAC model
        try:
            self.model = SAC.load(model_path)
            print("Dense Reward SAC model loaded successfully")
        except Exception as e:
            print(f"Error loading SAC model: {e}")
            raise

        # Initialize state processor with same configuration as training
        try:
            self.state_processor = StateProcessor(
                history_length=self.config.history_length,
                future_plan_length=self.config.future_plan_length,
                normalize=self.config.normalize_states,
                mode="mvp"
            )

            # Load normalization statistics
            self.state_processor.load_statistics()
            print(f"State processor initialized: {self.state_processor.expected_dim}D")

        except Exception as e:
            print(f"Warning: State processor initialization failed: {e}")
            # Fallback to basic processing
            self.state_processor = None

        # Track previous steering for continuity
        self.prev_steer = 0.0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """
        Update the controller and return the steering action.

        Args:
            target_lataccel: The target lateral acceleration
            current_lataccel: The current lateral acceleration
            state: The current state of the vehicle (named tuple with v_ego, a_ego, roll_lataccel)
            future_plan: The future plan for the next N frames

        Returns:
            float: The steering action to apply
        """
        try:
            if self.state_processor is not None:
                # Use the same state processing as training
                processed_state = self.state_processor.process_state(
                    target_lataccel=target_lataccel,
                    current_lataccel=current_lataccel,
                    state=state,
                    future_plan=future_plan
                )
            else:
                # Fallback: create simple state vector
                # Basic state: [target_lataccel, current_lataccel, v_ego, a_ego, roll_lataccel, prev_steer]
                processed_state = np.array([
                    target_lataccel,
                    current_lataccel,
                    getattr(state, 'v_ego', 0.0),
                    getattr(state, 'a_ego', 0.0),
                    getattr(state, 'roll_lataccel', 0.0),
                    self.prev_steer
                ])

                # Pad to expected dimension if needed
                if len(processed_state) < 20:
                    processed_state = np.pad(processed_state, (0, 20 - len(processed_state)))
                elif len(processed_state) > 20:
                    processed_state = processed_state[:20]

            # Get action from SAC model (deterministic)
            action, _ = self.model.predict(processed_state, deterministic=True)

            # Extract scalar action
            if hasattr(action, '__len__') and len(action) > 0:
                steer_action = float(action[0])
            else:
                steer_action = float(action)

            # Clip to valid steering range
            steer_action = np.clip(steer_action, -2.0, 2.0)

            # Update previous steering for next iteration
            self.prev_steer = steer_action

            return steer_action

        except Exception as e:
            print(f"Warning: SAC controller update failed: {e}")
            # Fallback to simple proportional control
            error = target_lataccel - current_lataccel
            fallback_action = np.clip(error * 0.3, -2.0, 2.0)
            self.prev_steer = fallback_action
            return fallback_action