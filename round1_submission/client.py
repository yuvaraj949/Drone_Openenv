from openenv.client import OpenEnvClient
from environment.models import Observation, Action, Reward

class DroneTrafficClient(OpenEnvClient[Observation, Action, Reward]):
    """
    Typed client for the Drone Traffic Control environment.
    Provides a standardized programmatic interface for the OpenEnv SDK.
    """
    pass
