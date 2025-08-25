from . import BaseController
import os
import numpy as np
import torch

try:
    from sac_controller.controller import SACController
    from sac_controller.utils.dummy_env import DummyEnv

    SAC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import SAC controller: {e}")
    print(
        "Make sure to install sac_controller package with: python -m pip install -e /path/to/control-drive"
    )
    SAC_AVAILABLE = False


class Controller(BaseController):
    """
    SAC (Soft Actor-Critic) controller for lateral vehicle control.

    This controller loads a pre-trained SAC model and uses it for control.
    """

    def __init__(self, model_path=None):
        """
        Initialize the SAC controller.

        Args:
            model_path: Path to the trained SAC model (optional)
        """
        if not SAC_AVAILABLE:
            raise ImportError("SAC controller dependencies not available")

        # Get model path from command line args or use default
        if model_path is None:
            import sys

            # Look for --sac_model argument in sys.argv
            model_name = "sac_controller_stable"  # default
            for i, arg in enumerate(sys.argv):
                if arg == "--sac_model" and i + 1 < len(sys.argv):
                    model_name = sys.argv[i + 1]
                    break

            sac_drive_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "sac-drive"
            )
            model_path = os.path.join(sac_drive_path, "experiments", "models", f"{model_name}.zip")
            print(f"Using SAC model: {model_name}")

        # Create a dummy environment for the SAC agent (needed for SB3)
        # State: [vEgo, aEgo, roll, prev_steer] = 4 dimensions
        dummy_env = DummyEnv(state_dim=4, action_dim=1)

        # Initialize the SAC controller with dummy env
        self.controller = SACController(agent_kwargs={"env": dummy_env})

        # Load the trained model if it exists
        if os.path.exists(model_path):
            print(f"Loading SAC model from: {model_path}")
            # Pass the path - the load method will handle the .zip extension
            self.controller.load(model_path)
            print("SAC model loaded successfully")
        else:
            print(f"Warning: No trained model found at {model_path}")
            print("Using untrained SAC controller")

        # Set to evaluation mode (deterministic actions)
        self.controller.eval()

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        """
        Update the controller and return the steering action.

        Args:
            target_lataccel: The target lateral acceleration
            current_lataccel: The current lateral acceleration
            state: The current state of the vehicle
            future_plan: The future plan for the next N frames

        Returns:
            float: The steering action to apply
        """
        return self.controller.update(
            target_lataccel=target_lataccel,
            current_lataccel=current_lataccel,
            state=state,
            future_plan=future_plan,
        )
