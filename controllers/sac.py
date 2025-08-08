from . import BaseController
import sys
import os
import numpy as np
import torch

# Add the control-drive project to the path
control_drive_path = os.path.join(os.path.dirname(__file__), '..', '..', 'control-drive')
sys.path.append(control_drive_path)

try:
    from sac_controller.controller import SACController
    from sac_controller.utils.dummy_env import DummyEnv
    SAC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import SAC controller: {e}")
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
        
        # Default model path if none provided
        if model_path is None:
            model_path = os.path.join(control_drive_path, "models", "sac_controller_final")
        
        # Initialize the SAC controller
        self.controller = SACController()
        
        # Load the trained model if it exists
        if os.path.exists(model_path + ".zip"):
            print(f"Loading SAC model from: {model_path}")
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
            future_plan=future_plan
        ) 