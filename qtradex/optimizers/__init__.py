from qtradex.optimizers.ipse import IPSE, IPSEoptions
from qtradex.optimizers.lsga import LSGA, LSGAoptions
try:
    from qtradex.optimizers.mouse_wheel_optimizer import MouseWheelTuner
except ImportError:
    class MouseWheelTuner:  # dummy — tkinter not available
        pass
from qtradex.optimizers.qpso import QPSO, QPSOoptions
from qtradex.optimizers.aion import AION, AIONoptions
from qtradex.optimizers.grid_search import GridSearch, GridSearchOptions
try:
    from qtradex.optimizers.rl import RLPPO, RLPPOoptions
except ImportError:
    class RLPPO:  # dummy — gymnasium/stable-baselines3 not installed
        pass
    class RLPPOoptions:
        pass
