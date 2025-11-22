"""
Evaluate a trained PPO agent on the KUKA joint control task.
"""

import argparse
import numpy as np
import torch
import time

from kuka_env import KukaJointControlEnv
from train_ppo import Agent
import gymnasium as gym


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/best_model.pt",
        help="path to the trained model")
    parser.add_argument("--num-episodes", type=int, default=10,
        help="number of episodes to evaluate")
    parser.add_argument("--render", action="store_true",
        help="whether to render the environment")
    parser.add_argument("--deterministic", action="store_true",
        help="whether to use deterministic actions (mean instead of sampling)")
    parser.add_argument("--seed", type=int, default=42,
        help="seed for evaluation")
    return parser.parse_args()


def evaluate(agent, env, num_episodes=10, deterministic=False, render=False):
    """Evaluate the agent for a given number of episodes."""
    episode_rewards = []
    episode_lengths = []
    success_count = 0

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        print(f"\n=== Episode {episode + 1}/{num_episodes} ===")

        while not done:
            with torch.no_grad():
                obs_tensor = torch.Tensor(obs).unsqueeze(0)

                if deterministic:
                    # Use mean action (deterministic)
                    action = agent.actor_mean(obs_tensor).cpu().numpy()[0]
                else:
                    # Sample from distribution
                    action, _, _, _ = agent.get_action_and_value(obs_tensor)
                    action = action.cpu().numpy()[0]

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

            if render:
                time.sleep(1.0 / 60.0)  # Slow down for visualization

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if info.get("is_success", False):
            success_count += 1
            print(f"SUCCESS! Reward: {episode_reward:.2f}, Length: {episode_length}, Distance: {info['distance']:.4f}")
        else:
            print(f"FAILED. Reward: {episode_reward:.2f}, Length: {episode_length}, Distance: {info['distance']:.4f}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Success Rate: {success_count}/{num_episodes} ({100 * success_count / num_episodes:.1f}%)")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print("=" * 60)

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "success_rate": success_count / num_episodes,
    }


if __name__ == "__main__":
    args = parse_args()

    # Set seed for torch (for reproducible policy behavior if using --deterministic)
    # Do NOT seed numpy so that environment targets are different each episode
    torch.manual_seed(args.seed)

    # Create environment
    render_mode = "human" if args.render else None
    env = KukaJointControlEnv(render_mode=render_mode)

    # Create a dummy vectorized env for Agent initialization
    dummy_env = gym.vector.SyncVectorEnv([lambda: env])

    # Load agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(dummy_env).to(device)

    # Load model weights
    try:
        agent.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Successfully loaded model from {args.model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
        print("Please train a model first using train_ppo.py")
        exit(1)

    agent.eval()

    # Run evaluation
    results = evaluate(
        agent=agent,
        env=env,
        num_episodes=args.num_episodes,
        deterministic=args.deterministic,
        render=args.render
    )

    env.close()
