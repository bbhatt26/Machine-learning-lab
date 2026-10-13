# Q21: Check libraries required for DQN
try:
    import torch
    print("PyTorch version:", torch.__version__)
except ImportError:
    print("PyTorch is not installed. Run: python -m pip install torch")
try:
    import gymnasium
    print("Gymnasium version:", gymnasium.__version__)
except ImportError:
    print("Gymnasium is not installed. Run: python -m pip install gymnasium")
