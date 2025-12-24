"""RecurrentPPO controller adapter for controls_challenge evaluation.

This controller loads a trained RecurrentPPO model and uses it for lateral control.
It implements the BaseController interface expected by tinyphysics.py.

CRITICAL: This controller maintains LSTM hidden states that persist across timesteps
within an episode. The hidden states must be properly managed.

Usage with eval.py:
    python eval.py --model_path models/tinyphysics.onnx \
                   --data_path data \
                   --test_controller recurrent_ppo \
                   --sac_model <experiment_name> \
                   --baseline_controller pid
"""

import sys
from pathlib import Path

import numpy as np

from . import BaseController

# Add project root to path for sac imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sb3_contrib import RecurrentPPO  # noqa: E402

from sac.features.state_processor import StateProcessor  # noqa: E402


class Controller(BaseController):
    """RecurrentPPO controller adapter for controls_challenge.

    Loads a trained RecurrentPPO model and processes observations using StateProcessor
    to produce steering commands compatible with the TinyPhysicsSimulator.

    CRITICAL: Maintains LSTM hidden states across timesteps for temporal reasoning.
    """

    def __init__(
        self,
        experiment_name: str = "baseline",
        checkpoint_step: int | None = None,
    ):
        """Initialize the RecurrentPPO controller.

        Args:
            experiment_name: Name of the experiment directory under experiments/,
                or "current" to use the promoted model.
            checkpoint_step: Specific checkpoint step to load. If None, loads
                final_model.zip or the latest checkpoint.

        Note:
            State processor configuration is loaded from the experiment's
            config.yaml to ensure consistency with training.
        """
        # Handle "current" alias - use promoted model
        if experiment_name == "current":
            model_path = PROJECT_ROOT / "models" / "current"
            if not model_path.exists():
                raise FileNotFoundError(
                    "No promoted model. Run 'sac registry promote' first."
                )
            # Resolve symlink to actual path
            model_path = model_path.resolve()
            experiments_dir = model_path.parent.parent
        else:
            # Find model checkpoint from experiment directory
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

        # Load RecurrentPPO model
        self.model = RecurrentPPO.load(model_path)

        # Load state processor config from experiment's config.yaml
        config_path = experiments_dir / "config.yaml"
        observation_history = 20
        use_future_plan = True
        future_plan_horizon = 50
        future_plan_mode = "raw"

        if config_path.exists():
            from sac.config import load_config

            config = load_config(config_path)
            observation_history = config.environment.observation_history
            use_future_plan = config.environment.use_future_plan
            future_plan_horizon = config.environment.future_plan_horizon
            future_plan_mode = config.environment.future_plan_mode

        # Initialize state processor with config from training
        self.state_processor = StateProcessor(
            history_length=observation_history,
            use_future_plan=use_future_plan,
            future_plan_horizon=future_plan_horizon,
            future_plan_mode=future_plan_mode,
        )

        # Track last action for state processor
        self.last_action = 0.0

        # LSTM hidden states - None signals fresh start
        self.lstm_states = None

        # Episode boundary tracking
        self.episode_start = True

        # Reset state processor
        self.state_processor.reset()

    def update(
        self,
        target_lataccel: float,
        current_lataccel: float,
        state,
        future_plan,
    ) -> float:
        """Get steering action from RecurrentPPO model.

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

        # RecurrentPPO requires batch dimension
        obs = processed.observation.reshape(1, -1)

        # Create episode_starts array with proper shape and dtype
        episode_starts = np.array([self.episode_start], dtype=bool)

        # Predict with LSTM state propagation
        action, self.lstm_states = self.model.predict(
            obs,
            state=self.lstm_states,
            episode_start=episode_starts,
            deterministic=True,
        )

        # After first step, no longer episode start
        self.episode_start = False

        # Extract scalar action (handles various array shapes)
        action_value = float(np.clip(np.atleast_1d(action).flatten()[0], -2.0, 2.0))

        self.last_action = action_value
        return action_value
