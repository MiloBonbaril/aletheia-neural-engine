# Modular Reinforcement Learning Engine

A highly modular, robust, and clean implementation of a Proximal Policy Optimization (PPO) pipeline in PyTorch. The codebase is designed to train agents in continuous action-space control problems, starting with the classic bipedal locomotion task: `BipedalWalker-v3`.

A primary architectural feature of this project is the **strict separation of the environment simulator engine from the RL agent and training pipeline**. This ensures that you can completely swap out the underlying game engine (e.g., from Gymnasium to a custom Pygame engine, physical simulator, WebGL interface, or a Rust-based FFI bridge) with minimal effort.

---

## 🛠️ File Structure

The project has been decomposed into modular single-responsibility components:

```
├── config.py          # Centralized hyperparameter and environment settings
├── env_interface.py   # Base abstract interface defining the unified environment API
├── gym_env.py         # Concrete implementation of the environment interface wrapping Gymnasium
├── networks.py        # PyTorch Actor and Critic Neural Networks for continuous control
├── buffer.py          # Vector-friendly rollout buffer storing experiences and computing GAE
├── agent.py           # PPO Agent handling updates, action sampling, and saving/loading
├── train.py           # Main training orchestrator, stats logger, and checkpoint saver
├── evaluate.py        # Independent evaluation script with deterministic execution and rendering
└── README.md          # Comprehensive setup, execution, and architectural guide
```

---

## 🔄 Switching Environments (Decoupled Design)

To implement a new game or completely replace Gymnasium, you do not need to modify the agent, buffer, networks, or training script. 

### 1. The Environment Boundary
Every interaction in the training loop goes through the `BaseEnvironment` abstract class defined in `env_interface.py`. This interface specifies what an environment must provide:

```python
class BaseEnvironment(ABC):
    @abstractmethod
    def reset(self, seed: int = None) -> Tuple[np.ndarray, Dict[str, Any]]: ...
    
    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: ...
    
    @abstractmethod
    def close(self) -> None: ...
    
    @property
    @abstractmethod
    def observation_shape(self) -> Tuple[int, ...]: ...
    
    @property
    @abstractmethod
    def action_shape(self) -> Tuple[int, ...]: ...
    
    @property
    @abstractmethod
    def action_low(self) -> np.ndarray: ...
    
    @property
    @abstractmethod
    def action_high(self) -> np.ndarray: ...
```

### 2. How to Swivel to a Custom Game Simulator
If you want to plug in a custom, non-Gymnasium game (for example, a physics engine written in pure Pygame or a C++/Rust simulation bridge):

1. **Create a new wrapper file** (e.g., `custom_game_env.py`).
2. **Subclass `BaseEnvironment`** and implement all abstract methods/properties to interact with your custom simulation.
3. **Update `train.py` / `evaluate.py`** to import and instantiate your new wrapper instead of `GymnasiumEnv`:
   ```python
   # Replace: from gym_env import GymnasiumEnv
   #          env = GymnasiumEnv(config.env_id)
   
   from custom_game_env import MyCustomGameEnv
   env = MyCustomGameEnv(config.env_id)
   ```

---

## 🚀 Getting Started

### 📋 Prerequisites

Install the required dependencies using pip. The environment requires `Box2D` to run the `BipedalWalker-v3` physics simulation.

```bash
# Install PyTorch, Gymnasium, swig, and the Box2D wrapper
pip install torch numpy swig gymnasium "gymnasium[box2d]" pygame
```

### 🏋️ Training the Agent

To start training the PPO agent from scratch, execute:

```bash
python train.py
```

* **Checkpoints**: The best-performing model (determined during periodic evaluation runs) is automatically saved to `checkpoints/ppo_bipedal_best.pt`.
* **Hyperparameters**: You can easily adjust learning rates, layer dims, and discount factor coefficients by editing `config.py`.

### 📊 Evaluating a Trained Agent

To evaluate the performance of your trained agent or watch it walk, run:

```bash
# Run deterministic evaluation over 5 episodes
python evaluate.py --checkpoint checkpoints/ppo_bipedal_best.pt --episodes 5

# Run with visual rendering (Pygame window)
python evaluate.py --checkpoint checkpoints/ppo_bipedal_best.pt --episodes 3 --render
```

---

## 🧠 Algorithmic Enhancements Included

To ensure that the PPO agent successfully solves the continuous joint-torque coordination in `BipedalWalker-v3`, we have integrated several stability improvements:

* **Orthogonal Layer Initialization**: Network weights are initialized using orthogonal matrices, preventing gradient explosions and accelerating convergence.
* **Actor Output Scaling**: The policy network outputs action values using a `Tanh` layer, naturally scaling the torque boundaries to the continuous range of `[-1, 1]` required by the walker joints.
* **State-Independent Learnable Action Std**: The exploration standard deviation is modeled as an independent vector parameter, allowing the policy to gradually fine-tune its noise levels as training progresses.
* **Advantage Normalization**: Advantage targets are normalized at the mini-batch level during updating, drastically stabilizing optimization steps.
* **Generalized Advantage Estimation (GAE)**: Balances variance and bias for robust reinforcement updates.
* **Linear Learning Rate Decay**: Slowly cools down the gradient steps as the agent approaches the step budget.
