import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data


class KukaJointControlEnv(gym.Env):
    """
    Gym environment for KUKA iiwa robot joint position control.

    Task: Control robot joints to reach a target joint configuration.

    Observation: [joint_positions (7), joint_velocities (7), target_joint_positions (7)] = 21-dim
    Action: Joint torques (7-dim, continuous)
    Reward: -distance to target configuration
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, max_steps=500):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0

        # Connect to PyBullet
        if render_mode == "human":
            self.client_id = p.connect(p.GUI)
        else:
            self.client_id = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Action space: joint torques for 7 joints
        # Normalized to [-1, 1], will be scaled to actual torque limits
        self.max_torque = 100.0  # Maximum torque per joint
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        # Observation space: [joint_pos (7), joint_vel (7), target_pos (7)]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32
        )

        # Joint limits for KUKA iiwa
        self.joint_lower_limits = np.array([-2.96, -2.09, -2.96, -2.09, -2.96, -2.09, -3.05])
        self.joint_upper_limits = np.array([2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.05])

        self.robot_id = None
        self.joint_indices = None
        self.target_joint_pos = None

        # Debug visualization IDs
        self.debug_items = []

        self._setup_scene()

    def _setup_scene(self):
        """Set up the PyBullet scene."""
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client_id)

        # Load plane
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)

        # Load KUKA robot
        start_pos = [0, 0, 0]
        start_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF(
            "kuka_iiwa/model.urdf", start_pos, start_orn,
            useFixedBase=True, physicsClientId=self.client_id
        )

        # Get controllable joint indices
        self.joint_indices = []
        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.client_id)
        for j in range(num_joints):
            info = p.getJointInfo(self.robot_id, j, physicsClientId=self.client_id)
            joint_type = info[2]
            if joint_type == p.JOINT_REVOLUTE:
                self.joint_indices.append(j)

        # Disable default motor control to use torque control
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.joint_indices,
            controlMode=p.VELOCITY_CONTROL,
            forces=[0] * len(self.joint_indices),
            physicsClientId=self.client_id
        )
    def _sample_target_joint_pos(self):
        """Sample a random target joint configuration within limits."""
        return np.random.uniform(
            self.joint_lower_limits,
            self.joint_upper_limits
        ).astype(np.float32)

    def _get_ee_position(self, joint_positions=None):
        """Get end-effector position for given joint configuration (or current if None)."""
        if joint_positions is not None:
            # Temporarily set joints to get EE position
            current_states = p.getJointStates(
                self.robot_id, self.joint_indices, physicsClientId=self.client_id
            )
            current_pos = [state[0] for state in current_states]

            # Set to target configuration
            for idx, pos in zip(self.joint_indices, joint_positions):
                p.resetJointState(
                    self.robot_id, idx, pos, targetVelocity=0.0,
                    physicsClientId=self.client_id
                )

            # Get EE position
            ee_link_index = p.getNumJoints(self.robot_id, physicsClientId=self.client_id) - 1
            state = p.getLinkState(self.robot_id, ee_link_index, computeForwardKinematics=1, physicsClientId=self.client_id)
            ee_pos = np.array(state[0])

            # Restore original configuration
            for idx, pos in zip(self.joint_indices, current_pos):
                p.resetJointState(
                    self.robot_id, idx, pos, targetVelocity=0.0,
                    physicsClientId=self.client_id
                )

            return ee_pos
        else:
            # Get current EE position
            ee_link_index = p.getNumJoints(self.robot_id, physicsClientId=self.client_id) - 1
            state = p.getLinkState(self.robot_id, ee_link_index, computeForwardKinematics=1, physicsClientId=self.client_id)
            return np.array(state[0])

    def _clear_debug_items(self):
        """Clear all debug visualization items."""
        for item_id in self.debug_items:
            p.removeUserDebugItem(item_id, physicsClientId=self.client_id)
        self.debug_items = []

    def _add_debug_visualization(self, init_joint_pos):
        """Add debug visualization for start and target configurations."""
        if self.render_mode != "human":
            return  # Only visualize in GUI mode

        # Clear previous debug items
        self._clear_debug_items()

        # Get current (start) EE position
        start_ee_pos = self._get_ee_position()

        # Get target EE position
        target_ee_pos = self._get_ee_position(self.target_joint_pos)
        
        start_sphere = p.addUserDebugLine(
            start_ee_pos,
            start_ee_pos,
            lineColorRGB=[0, 1, 0],
            lineWidth=10,
            physicsClientId=self.client_id
        )

        target_sphere = p.addUserDebugLine(
            target_ee_pos,
            target_ee_pos,
            lineColorRGB=[1, 0, 0],
            lineWidth=10,
            physicsClientId=self.client_id
        )
        self.debug_items.append(target_sphere)

        # Draw line connecting start to target
        line = p.addUserDebugLine(
            start_ee_pos,
            target_ee_pos,
            lineColorRGB=[1, 1, 0],
            lineWidth=2,
            physicsClientId=self.client_id
        )
        self.debug_items.append(line)

        distance = np.linalg.norm(target_ee_pos - start_ee_pos)
        mid_point = (start_ee_pos + target_ee_pos) / 2
        dist_text = p.addUserDebugText(
            f"Distance: {distance:.3f}m",
            mid_point,
            textColorRGB=[1, 1, 0],
            textSize=1.2,
            physicsClientId=self.client_id
        )
        self.debug_items.append(dist_text)

    def _get_obs(self):
        """Get current observation."""
        joint_states = p.getJointStates(
            self.robot_id, self.joint_indices, physicsClientId=self.client_id
        )

        joint_positions = np.array([state[0] for state in joint_states], dtype=np.float32)
        joint_velocities = np.array([state[1] for state in joint_states], dtype=np.float32)

        obs = np.concatenate([joint_positions, joint_velocities, self.target_joint_pos])
        return obs

    def _get_reward(self, obs):
        """Calculate reward based on distance to target."""
        joint_positions = obs[:7]
        # Negative L2 distance to target
        distance = np.linalg.norm(joint_positions - self.target_joint_pos)
        reward = -distance

        # Small bonus for being very close
        if distance < 0.1:
            reward += 1.0

        return reward

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0

        # Sample new target configuration
        self.target_joint_pos = self._sample_target_joint_pos()

        # Reset robot to random initial configuration
        init_joint_pos = self._sample_target_joint_pos()
        for idx, pos in zip(self.joint_indices, init_joint_pos):
            p.resetJointState(
                self.robot_id, idx, pos, targetVelocity=0.0,
                physicsClientId=self.client_id
            )

        # Add debug visualization (only in GUI mode)
        self._add_debug_visualization(init_joint_pos)

        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action):
        """Execute one step in the environment."""
        self.current_step += 1

        # Scale action to actual torque limits
        torques = action * self.max_torque

        # Apply torques
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=self.joint_indices,
            controlMode=p.TORQUE_CONTROL,
            forces=torques,
            physicsClientId=self.client_id
        )

        # Step simulation
        p.stepSimulation(physicsClientId=self.client_id)

        # Get observation and reward
        obs = self._get_obs()
        reward = self._get_reward(obs)

        # Check termination
        distance = np.linalg.norm(obs[:7] - self.target_joint_pos)
        terminated = distance < 0.05  # Success threshold
        truncated = self.current_step >= self.max_steps

        info = {
            "distance": distance,
            "is_success": terminated
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render is handled by PyBullet GUI."""
        if self.render_mode == "rgb_array":
            # Get camera image
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0, 0, 0],
                distance=2.0,
                yaw=45,
                pitch=-30,
                roll=0,
                upAxisIndex=2
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=1.0, nearVal=0.1, farVal=100.0
            )
            (_, _, px, _, _) = p.getCameraImage(
                width=640, height=480,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
                physicsClientId=self.client_id
            )
            rgb_array = np.array(px, dtype=np.uint8)
            rgb_array = rgb_array[:, :, :3]
            return rgb_array

    def close(self):
        """Clean up."""
        if self.client_id >= 0:
            self._clear_debug_items()
            p.disconnect(physicsClientId=self.client_id)


if __name__ == "__main__":
    # Test the environment
    env = KukaJointControlEnv(render_mode="human")

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    for i in range(1000):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)

        if i % 100 == 0:
            print(f"Step {i}: Reward={reward:.3f}, Distance={info['distance']:.3f}")

        if terminated or truncated:
            print(f"Episode finished at step {i}. Success: {info['is_success']}")
            obs, info = env.reset()

    env.close()
