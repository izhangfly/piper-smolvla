"""PiPER SmolVLA: fine-tune and run LeRobot's SmolVLA on an AgileX PiPER arm.

The robot class is exported lazily, so the command line (and the GPU box's training supervisor) can import
this package without loading LeRobot. LeRobot finds the robot class for --robot.type=piper_http here: it
looks for `PiperHttp` in the package that defines `PiperHttpConfig`'s module.
"""
__version__ = "0.1.0"
__all__ = ["PiperHttp", "PiperHttpConfig", "__version__"]


def __getattr__(name):
    if name == "PiperHttp":
        from .robot import PiperHttp
        return PiperHttp
    if name == "PiperHttpConfig":
        from .config import PiperHttpConfig
        return PiperHttpConfig
    raise AttributeError(name)
