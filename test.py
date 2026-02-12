"""Run the game environment with random actions."""

import argparse
import time
from env import GameEnv, GameEnvConfig


def main(args):
    # Create environment
    config = GameEnvConfig(
        render_mode=args.render_mode,
        max_steps=args.max_steps,
    )
    env = GameEnv(config=config)
    
    print("=" * 60)
    print("Running Game Environment with Random Actions")
    print("=" * 60)
    print(f"Render Mode: {args.render_mode}")
    print(f"Num Episodes: {args.num_episodes}")
    print(f"Max Steps: {args.max_steps}")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print()

    # Run multiple episodes
    t_start = time.time()
    n_frames = 0
    for episode in range(args.num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        
        print(f"\n--- Episode {episode + 1} ---")
        print(f"Initial Energy: {obs['energy'][0]:.1f}")
        
        done = False
        while not done:
            # Sample random action
            action = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            step_count += 1
            n_frames += 1
            
            # Print progress every 100 steps
            if step_count % 100 == 0:
                print(f"  Step {step_count}: Energy={obs['energy'][0]:.1f}, Episode return={episode_reward:.2f}")
            
            done = terminated or truncated
        
        # Episode summary
        reason = info.get("reason", "Truncated")
        print(f"\nEpisode {episode + 1} finished:")
        print(f"  Reason: {reason}")
        print(f"  Steps: {step_count}")
        print(f"  Episode return: {episode_reward:.2f}")
        print(f"  Final Energy: {obs['energy'][0]:.1f}")

    t_end = time.time()
    print("\n" + "=" * 60)
    print("Testing complete!")
    print(f"Total frames: {n_frames}")
    print(f"Total time: {t_end - t_start:.2f} seconds")
    print(f"FPS: {n_frames / (t_end - t_start):.2f}")
    print("=" * 60)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the game environment with random actions")
    parser.add_argument(
        "-n",
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to run (default: 5)"
    )
    parser.add_argument(
        "-rm",
        "--render_mode",
        type=str,
        default=None,
        choices=[None, "human", "rgb_array"],
        help="Render mode: None (headless), 'human', or 'rgb_array' (default: None)"
    )
    parser.add_argument(
        "-ms",
        "--max_steps",
        type=int,
        default=1000,
        help="Maximum steps per episode (default: 1000)"
    )
    args = parser.parse_args()

    main(args)
