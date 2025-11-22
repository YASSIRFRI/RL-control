# KUKA Robot RL Training with PPO

This project implements Proximal Policy Optimization (PPO) for training a KUKA iiwa robot to perform joint position control tasks.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

Train the PPO agent:

```bash
python train_ppo.py
```

### Training Options

```bash
python train_ppo.py --total-timesteps 1000000 --learning-rate 3e-4 --num-steps 2048
```

Key arguments:
- `--total-timesteps`: Total training timesteps (default: 1,000,000)
- `--learning-rate`: Learning rate (default: 3e-4)
- `--num-steps`: Steps per rollout (default: 2048)
- `--update-epochs`: PPO update epochs (default: 10)
- `--gamma`: Discount factor (default: 0.99)
- `--gae-lambda`: GAE lambda (default: 0.95)
- `--clip-coef`: PPO clip coefficient (default: 0.2)
- `--max-episode-steps`: Max steps per episode (default: 500)
- `--resume-from`: Resume training from a checkpoint (e.g., `models/checkpoint_100.pt` or `models/best_model.pt`)

### Resuming Training

To continue training from where you left off:

```bash
# Resume from the best model
python train_ppo.py --resume-from models/best_model.pt

# Resume from a specific checkpoint
python train_ppo.py --resume-from models/checkpoint_100.pt

# Resume and train for additional timesteps
python train_ppo.py --resume-from models/best_model.pt --total-timesteps 2000000
```

The training script will:
- Save the best model to `models/best_model.pt`
- Save periodic checkpoints to `models/checkpoint_<update>.pt`
- Save the final model to `models/final_model.pt`

## Evaluation

Evaluate a trained model:

```bash
python evaluate.py --model-path models/best_model.pt --num-episodes 10 --render
```

Options:
- `--model-path`: Path to trained model (default: models/best_model.pt)
- `--num-episodes`: Number of evaluation episodes (default: 10)
- `--render`: Show PyBullet GUI visualization
- `--deterministic`: Use deterministic actions (mean instead of sampling)

### Visualization

When using `--render`, you'll see:
- **Green marker**: START position (end-effector at initial configuration)
- **Red marker**: TARGET position (end-effector at goal configuration)
- **Yellow line**: Connection between start and target
- **Distance text**: Distance in meters between start and target

This helps you visualize where the robot needs to move its end-effector to reach the target joint configuration.

## Environment Details

**Task**: Control the KUKA iiwa robot's 7 joints to reach target joint configurations.

**Observation Space** (21-dim):
- Joint positions (7)
- Joint velocities (7)
- Target joint positions (7)

**Action Space** (7-dim):
- Joint torques (continuous, normalized to [-1, 1])

**Reward Function**:
- Negative L2 distance to target configuration
- Bonus of +1.0 when distance < 0.1

**Episode Termination**:
- Success: Distance to target < 0.05
- Max steps: 500 steps

## Testing the Environment

Test the environment with random actions:

```bash
python kuka_env.py
```

This will run the environment with random actions and display statistics.

## Files

- `kuka_env.py`: Gymnasium environment for KUKA robot
- `train_ppo.py`: PPO training script (CleanRL style)
- `evaluate.py`: Evaluation script for trained models
- `main.py`: Original demo scripts (model-based control, NL→LTL, etc.)
- `requirements.txt`: Python dependencies

## Tips

1. **Start with shorter training**: Try `--total-timesteps 100000` first to verify everything works
2. **Monitor progress**: Check the console output for episodic returns and success rate
3. **Tune hyperparameters**: If learning is slow, try adjusting:
   - Increase `--learning-rate` to 5e-4 for faster learning
   - Decrease `--num-steps` to 1024 for more frequent updates
   - Adjust `--max-torque` in kuka_env.py if the robot is too weak/strong

4. **GPU acceleration**: Training will automatically use CUDA if available (much faster)

## Expected Performance

With default hyperparameters, you should see:
- Initial episodes: Large negative rewards (-10 to -5)
- After ~200k steps: Rewards improving to -2 to 0
- After ~500k steps: Occasional successes, rewards approaching 0
- After ~1M steps: Consistent successes with high success rate

Training time: ~30-60 minutes on a modern CPU, ~10-20 minutes with GPU.
