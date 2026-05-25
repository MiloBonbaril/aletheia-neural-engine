import random
import sys

# Monkeypatch random.shuffle to support the deprecated 'random' parameter removed in Python 3.11+
_orig_shuffle = random.shuffle
def _patched_shuffle(x, random_fn=None):
    if random_fn is not None:
        _int = int
        for i in reversed(range(1, len(x))):
            j = _int(random_fn() * (i + 1))
            x[i], x[j] = x[j], x[i]
    else:
        _orig_shuffle(x)
random.shuffle = _patched_shuffle

# Parse absl flags before importing pysc2
from absl import flags
FLAGS = flags.FLAGS
# Avoid raising an error if flags have already been parsed
try:
    FLAGS(sys.argv)
except flags.CantUseIpythonError:
    pass

from pysc2.env import sc2_env
from pysc2.lib import features
from pysc2.lib import actions
from pysc2.maps import lib

class CustomMeleeMap(lib.Map):
    """Custom Map class referencing an installed melee map."""
    directory = ""
    filename = "AcropolisLE"
    players = 2

def main():
        print("--------------------------------------------------")
    print("PySC2: All Possible Actions in the Action Space:")
    print("--------------------------------------------------")
    for action in actions.FUNCTIONS:
        print(f"Action ID: {action.id:<4} Name: {action.name:<35} Args: {action.args}")
    print(f"Total possible actions: {len(actions.FUNCTIONS)}")
    print("--------------------------------------------------\n")

    print("Initializing StarCraft II Environment...")
    env = sc2_env.SC2Env(
        map_name=CustomMeleeMap(),
        players=[
            sc2_env.Agent(sc2_env.Race.zerg),
            sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)
        ],
        agent_interface_format=features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=84, minimap=64),
            use_feature_units=True
        ),
        step_mul=8,
        visualize=False
    )

    try:
        print("\nResetting environment...")
        timesteps = env.reset()
        print("Environment reset successful!")

        # A timestep is returned for each agent
        agent_timestep = timesteps[0]
        
        print("\n--------------------------------------------------")
        print("Observation structure & sample data:")
        print("--------------------------------------------------")
        print(f"Step Type: {agent_timestep.step_type}")
        print(f"Reward: {agent_timestep.reward}")
        print(f"Discount: {agent_timestep.discount}")
        
        obs = agent_timestep.observation
        print("\nKeys in observation dict:")
        for k, v in sorted(obs.items()):
            if isinstance(v, (list, tuple)):
                print(f"  - {k}: list/tuple of length {len(v)}")
            elif hasattr(v, "shape"):
                print(f"  - {k}: array with shape {v.shape}")
            else:
                print(f"  - {k}: {type(v).__name__} = {v}")

        print("\nPlayer stats:")
        print(f"  - Player ID: {obs.player.player_id}")
        print(f"  - Minerals: {obs.player.minerals}")
        print(f"  - Vespene: {obs.player.vespene}")
        print(f"  - Food used: {obs.player.food_used}")
        print(f"  - Food cap: {obs.player.food_cap}")
        print(f"  - Idle worker count: {obs.player.idle_worker_count}")
        print(f"  - Army count: {obs.player.army_count}")

        print("\nCurrently Available Actions (IDs and names) at Step 0:")
        for action_id in obs.available_actions:
            action_func = actions.FUNCTIONS[action_id]
            print(f"  - ID: {action_id:<4} Name: {action_func.name}")

    finally:
        print("\nClosing environment...")
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main()