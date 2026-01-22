# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Script to run an environment with zero action agent."""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pace agent for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Pace-Anymal-D-v0", help="Name of the task.")
parser.add_argument("--duration", type=float, default=20.0, help="Duration in seconds.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import pace_sim2real.tasks  # noqa: F401

def main():
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # reset environment to initialize buffers
    env.reset()

    articulation = env.unwrapped.scene["robot"]
    joint_order = env_cfg.sim2real.joint_order
    joint_ids = torch.tensor([articulation.joint_names.index(name) for name in joint_order], device=env.unwrapped.device)

    # Calculate timing
    sample_rate = 1 / env.unwrapped.sim.get_physics_dt()
    num_steps = int(args_cli.duration * sample_rate)
    counter = 0

    # Print Default Positions for verification
    default_pos = articulation.data.default_joint_pos[0, joint_ids]
    print("\n" + "="*50)
    print(f"{'Joint Name':<25} | {'Default Pos (Rad)':<15}")
    print("-" * 50)
    for i, name in enumerate(joint_order):
        print(f"{name:<25} | {default_pos[i]:>15.4f}")
    print("="*50 + "\n")

    # Static actions: zeros keep the robot at default_joint_pos
    static_actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    print(f"[INFO]: Simulation running for {args_cli.duration} seconds. Holding pose...")

    while simulation_app.is_running():
        with torch.inference_mode():
            # Apply zero actions to maintain the default standing/sitting pose
            env.step(static_actions)
            
            if counter % 400 == 0:
                print(f"[INFO]: Step {counter/sample_rate:.2f} seconds - Holding Pose")

            counter += 1
            if counter >= num_steps:
                break

    # close the simulator
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()