#!/usr/bin/env python3
"""
SAC Controller for controls_challenge evaluation using SACController.

This controller uses the SACController class from sac-drive which includes
optimized post-processing (smoothing + rate limiting) for improved performance.
"""

from . import BaseController
import sys
from pathlib import Path

# Add sac-drive to path
sac_drive_path = Path(__file__).parent.parent.parent / "sac-drive"
sys.path.insert(0, str(sac_drive_path))


def find_experiment_checkpoint(sac_drive_path, experiment_name=None, checkpoint_step=None):
    """
    Find experiment checkpoint by name and optional step.

    Args:
        sac_drive_path: Path to sac-drive project root
        experiment_name: Optional experiment name to load
        checkpoint_step: Optional specific checkpoint step number

    Returns:
        Path to checkpoint file, or None if not found
    """
    experiments_dir = sac_drive_path / "experiments"
    if not experiments_dir.exists():
        return None

    if experiment_name:
        exp_dir = experiments_dir / experiment_name
        if exp_dir.exists() and exp_dir.is_dir():
            checkpoints_dir = exp_dir / "checkpoints"
            if checkpoints_dir.exists():
                if checkpoint_step:
                    # Look for specific checkpoint
                    checkpoint = checkpoints_dir / f"checkpoint_step_{checkpoint_step}.zip"
                    if checkpoint.exists():
                        return str(checkpoint)
                else:
                    # Find latest checkpoint
                    checkpoints = list(checkpoints_dir.glob("checkpoint_step_*.zip"))
                    if checkpoints:
                        latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
                        return str(latest)
    else:
        # Find latest experiment with checkpoints
        valid_experiments = []
        for exp_dir in experiments_dir.iterdir():
            if exp_dir.is_dir():
                checkpoints_dir = exp_dir / "checkpoints"
                if checkpoints_dir.exists():
                    checkpoints = list(checkpoints_dir.glob("checkpoint_step_*.zip"))
                    if checkpoints:
                        latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
                        valid_experiments.append((exp_dir.stat().st_mtime, latest_checkpoint))

        if valid_experiments:
            # Return checkpoint from most recent experiment
            _, checkpoint = max(valid_experiments)
            return str(checkpoint)

    return None


try:
    from src.controller import SACController
    SAC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import SACController: {e}")
    SAC_AVAILABLE = False


class Controller(BaseController):
    """
    SAC controller wrapper using SACController with optimized post-processing.

    This implementation uses the SACController class which includes:
    - Sophisticated state processing
    - SAC model prediction
    - Optimized action smoothing (default: 0.95)
    - Action rate limiting (default: 1.17)
    - Action clipping to valid range
    """

    def __init__(
        self,
        model_path=None,
        experiment_name=None,
        smoothing_factor=None,
        max_action_change=None,
    ):
        """
        Initialize the SAC controller.

        Args:
            model_path: Optional direct path to model checkpoint file.
                       If None, will auto-discover from experiment_name or find latest.
            experiment_name: Optional experiment name to load.
                           If None, loads latest experiment with checkpoints.
            smoothing_factor: Optional smoothing factor override (0.0-1.0).
                            If None, uses SACController default (0.95).
            max_action_change: Optional max action change per timestep (rate limiting).
                             If None, uses SACController default (1.17).

        Raises:
            ImportError: If SAC dependencies not available
            FileNotFoundError: If no valid checkpoint found
        """
        if not SAC_AVAILABLE:
            raise ImportError("SACController dependencies not available")

        # Discover model path if not provided
        if model_path is None:
            discovered_path = find_experiment_checkpoint(sac_drive_path, experiment_name)
            if discovered_path is None:
                if experiment_name:
                    raise FileNotFoundError(
                        f"Experiment '{experiment_name}' not found or has no checkpoints."
                    )
                else:
                    raise FileNotFoundError(
                        "No experiments found with checkpoints. Please train a model first."
                    )
            model_path = discovered_path

        experiment_info = f" (experiment: {experiment_name})" if experiment_name else ""
        print(f"Loading SAC controller from: {model_path}{experiment_info}")

        # Initialize SACController with optimized post-processing
        kwargs = {"model_path": model_path}
        if smoothing_factor is not None:
            kwargs["smoothing_factor"] = smoothing_factor
            print(f"Using custom smoothing_factor: {smoothing_factor}")
        else:
            print("Using default smoothing_factor (0.95)")

        if max_action_change is not None:
            kwargs["max_action_change"] = max_action_change
            print(f"Using custom max_action_change: {max_action_change}")
        else:
            print("Using default max_action_change (1.17)")

        self.sac_controller = SACController(**kwargs)

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """
        Update the controller and return the steering action.

        This delegates to SACController.update() which applies the full
        post-processing pipeline including optimized smoothing.

        Args:
            target_lataccel: The target lateral acceleration
            current_lataccel: The current lateral acceleration
            state: The current state of the vehicle (namedtuple)
            future_plan: The future plan for the next N frames

        Returns:
            float: The steering action to apply [-2.0, 2.0]
        """
        return self.sac_controller.update(
            target_lataccel=target_lataccel,
            current_lataccel=current_lataccel,
            state=state,
            future_plan=future_plan
        )

    def reset_state(self):
        """
        Reset controller and state processor to initial state.

        Ensures consistent behavior between rollouts by clearing
        accumulated history and internal state.
        """
        self.sac_controller.reset()
