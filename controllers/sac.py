"""SAC controller adapter for controls_challenge evaluation.

This controller loads a trained SAC model and uses it for lateral control.
It implements the BaseController interface expected by tinyphysics.py.

Usage with eval.py:
    python eval.py --model_path models/tinyphysics.onnx \\
                   --data_path data \\
                   --test_controller sac \\
                   --sac_model <experiment_name> \\
                   --baseline_controller pid
"""

import sys
from pathlib import Path

import numpy as np

from . import BaseController

# Add project root to path for sac imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import SAC  # noqa: E402

from sac.features.state_processor import StateProcessor  # noqa: E402


class Controller(BaseController):
    """SAC controller adapter for controls_challenge.

    Loads a trained SAC model and processes observations using StateProcessor
    to produce steering commands compatible with the TinyPhysicsSimulator.
    """

    def __init__(
        self,
        experiment_name: str = "baseline",
        checkpoint_step: int | None = None,
        observation_history: int = 20,
        use_future_plan: bool = True,
        future_plan_horizon: int = 50,
    ):
        """Initialize the SAC controller.

        Args:
            experiment_name: Name of the experiment directory under experiments/.
            checkpoint_step: Specific checkpoint step to load. If None, loads
                final_model.zip or the latest checkpoint.
            observation_history: History length for state processor.
            use_future_plan: Whether to include future plan in observations.
            future_plan_horizon: Number of future timesteps in observation.
        """
        # Find model checkpoint
        experiments_dir = PROJECT_ROOT / "experiments" / experiment_name
        checkpoints_dir = experiments_dir / "checkpoints"

        if checkpoint_step is not None:
            model_path = checkpoints_dir / f"checkpoint_step_{checkpoint_step}.zip"
        else:
            # Try final_model first, then latest checkpoint
            model_path = checkpoints_dir / "final_model.zip"
            if not model_path.exists():
                # Find latest checkpoint
                checkpoints = sorted(checkpoints_dir.glob("checkpoint_step_*.zip"))
                if checkpoints:
                    model_path = checkpoints[-1]

        if not model_path.exists():
            raise FileNotFoundError(
                f"No model found for experiment '{experiment_name}'. "
                f"Checked: {model_path}"
            )

        # Load model
        self.model = SAC.load(model_path)

        # Initialize state processor with same config as training
        self.state_processor = StateProcessor(
            history_length=observation_history,
            use_future_plan=use_future_plan,
            future_plan_horizon=future_plan_horizon,
        )

        # Track last action for state processor
        self.last_action = 0.0

        # Reset state processor
        self.state_processor.reset()

    def update(
        self,
        target_lataccel: float,
        current_lataccel: float,
        state,
        future_plan,
    ) -> float:
        """Get steering action from SAC model.

        Args:
            target_lataccel: Target lateral acceleration.
            current_lataccel: Current lateral acceleration.
            state: State namedtuple (roll_lataccel, v_ego, a_ego).
            future_plan: FuturePlan namedtuple with future trajectory.

        Returns:
            Steering action in range [-2, 2].
        """
        # Process state into observation
        processed = self.state_processor.process(
            target_lataccel=target_lataccel,
            current_lataccel=current_lataccel,
            state=state,
            future_plan=future_plan,
            last_action=self.last_action,
        )

        # Get action from model (deterministic for evaluation)
        # Use atleast_1d().flatten() for robust action extraction
        action, _ = self.model.predict(processed.observation, deterministic=True)
        action_value = float(np.clip(np.atleast_1d(action).flatten()[0], -2.0, 2.0))

        self.last_action = action_value
        return action_value
