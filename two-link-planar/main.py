import pybullet as p
import pybullet_data
import time
import numpy as np


def connect_gui():
    cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    return cid

def load_kuka():
    start_pos = [0, 0, 0]
    start_orn = p.getQuaternionFromEuler([0, 0, 0])
    robot_id = p.loadURDF("kuka_iiwa/model.urdf", start_pos, start_orn, useFixedBase=True)
    return robot_id

def get_kuka_joint_indices(robot_id):
    num_joints = p.getNumJoints(robot_id)
    # KUKA iiwa: 7 revolute joints (0..6)
    controllable = []
    for j in range(num_joints):
        info = p.getJointInfo(robot_id, j)
        joint_type = info[2]
        if joint_type == p.JOINT_REVOLUTE:
            controllable.append(j)
    return controllable

def step_sim(dt=1./240.):
    p.stepSimulation()
    time.sleep(dt)

def get_end_effector_pos(robot_id, ee_link_index=None):
    if ee_link_index is None:
        ee_link_index = p.getNumJoints(robot_id) - 1
    state = p.getLinkState(robot_id, ee_link_index)
    return np.array(state[0])


# ==============================
#  OBJECTIVE 1: CONTROL
#  - Joint-space trajectory tracking (model-based via PyBullet)
# ==============================

def demo_objective1_model_based_control():
    print("\n=== Objective 1: Model-based joint trajectory tracking ===")
    connect_gui()
    robot_id = load_kuka()
    joint_indices = get_kuka_joint_indices(robot_id)

    # Initial and target joint configuration
    q_init = np.zeros(len(joint_indices))
    q_target = np.array([0.5, -0.5, 0.3, -1.0, 0.4, 0.7, -0.3])

    # Reset to initial
    for idx, q in zip(joint_indices, q_init):
        p.resetJointState(robot_id, idx, q)

    T = 5.0
    dt = 1. / 240.
    steps = int(T / dt)

    for k in range(steps):
        alpha = k / steps
        q_ref = (1 - alpha) * q_init + alpha * q_target

        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=q_ref.tolist(),
            forces=[200]*len(joint_indices)
        )
        step_sim(dt)

    print("Finished Objective 1. Robot should have moved smoothly to target joint configuration.")


# ==============================
#  OBJECTIVE 2: NATURAL LANGUAGE → LTL → CONTROL
# ==============================

# 1) Darija → English (simple dictionary)
def translate_darija_to_english(cmd: str) -> str:
    cmd = cmd.lower().strip()
    dict_darija = {
        "sir l limin": "move the end-effector to the right",
        "sir l lisar": "move the end-effector to the left",
        "tjanab l hait": "avoid the wall",
        "waqef hna": "stop here"
    }
    return dict_darija.get(cmd, "unknown command")

# 2) English → LTL (very simplified)
def english_to_ltl(eng: str) -> str:
    eng = eng.lower()
    if "to the right" in eng:
        return "F region_right"   # eventually enter region_right
    if "to the left" in eng:
        return "F region_left"
    if "avoid the wall" in eng:
        return "G ! region_wall"  # always avoid region_wall
    return "True"

# 3) LTL → goal region (here: a Cartesian goal)
def ltl_to_cartesian_goal(ltl_formula: str):
    # We choose arbitrary reachable positions as "regions"
    if "region_right" in ltl_formula:
        return np.array([0.7, -0.3, 0.7])  # x,y,z in world frame
    if "region_left" in ltl_formula:
        return np.array([0.7, 0.3, 0.7])
    # default "do nothing"
    return np.array([0.7, 0.0, 0.7])

# 4) Controller that uses IK (model-based)
def generate_ik_controller(robot_id, joint_indices, target_pos, ee_link_index=None):
    if ee_link_index is None:
        ee_link_index = p.getNumJoints(robot_id) - 1

    def controller_step():
        joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=robot_id,
            endEffectorLinkIndex=ee_link_index,
            targetPosition=target_pos.tolist()
        )
        joint_positions = [joint_positions[j] for j in range(len(joint_indices))]
        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=joint_positions,
            forces=[200]*len(joint_indices)
        )
    return controller_step

def demo_objective2_nl_to_ltl():
    print("\n=== Objective 2: Natural language → LTL → controller → motion ===")
    connect_gui()
    robot_id = load_kuka()
    joint_indices = get_kuka_joint_indices(robot_id)

    # Reset joints to zero
    for idx in joint_indices:
        p.resetJointState(robot_id, idx, 0.0)

    # Example user command in Darija
    user_cmd = "sir l limin u tjanab l hait"

    # For demo we consider main part "sir l limin"
    eng = translate_darija_to_english("sir l limin")
    ltl = english_to_ltl(eng)
    goal_pos = ltl_to_cartesian_goal(ltl)

    print(f"Darija command: {user_cmd}")
    print(f"English: {eng}")
    print(f"LTL: {ltl}")
    print(f"Goal position (cartesian): {goal_pos}")

    controller_step = generate_ik_controller(robot_id, joint_indices, goal_pos)

    T = 5.0
    dt = 1. / 240.
    steps = int(T / dt)

    # visualize goal as small debug marker (sphere)
    p.addUserDebugLine(goal_pos, goal_pos + np.array([0, 0, 0.1]), [1, 0, 0], 3)

    for _ in range(steps):
        controller_step()
        step_sim(dt)

    final_pos = get_end_effector_pos(robot_id)
    print("Final end-effector position:", final_pos)
    print("Norm error to goal:", np.linalg.norm(final_pos - goal_pos))


# ==============================
#  OBJECTIVE 3: MODEL-BASED vs DATA-DRIVEN
# ==============================

class DataDrivenIKApprox:
    """
    Fake 'data-driven' controller:
    - In reality you'd train a NN on (x,y,z) → joint angles.
    - Here we approximate by:
        q = W * target_pos + b  (simple linear model)
    """
    def __init__(self, W=None, b=None):
        if W is None:
            self.W = np.array([
                [ 1.0,  0.0,  0.0],
                [ 0.0,  1.0,  0.0],
                [ 0.2, -0.2,  0.0],
                [-0.3,  0.0,  0.2],
                [ 0.0, -0.3,  0.2],
                [ 0.3,  0.3, -0.4],
                [-0.2,  0.2, -0.2],
            ])
        else:
            self.W = W
        if b is None:
            self.b = np.zeros(7)
        else:
            self.b = b

    def predict_joints(self, target_pos):
        q = self.W @ target_pos + self.b
        return q

def demo_objective3_model_vs_data_driven():
    print("\n=== Objective 3: Model-based IK vs Data-driven IK approximation ===")
    connect_gui()
    robot_id = load_kuka()
    joint_indices = get_kuka_joint_indices(robot_id)
    ee_index = p.getNumJoints(robot_id) - 1

    # Reset joints to zero
    for idx in joint_indices:
        p.resetJointState(robot_id, idx, 0.0)

    # Choose the same goal for both controllers
    target_pos = np.array([0.7, -0.2, 0.7])
    p.addUserDebugLine(target_pos, target_pos + np.array([0, 0, 0.1]), [0, 1, 0], 3)

    # --- 1) Model-based (PyBullet IK) ---
    print("Running model-based (analytical IK via PyBullet)...")
    T = 4.0
    dt = 1. / 240.
    steps = int(T / dt)

    for _ in range(steps):
        joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=robot_id,
            endEffectorLinkIndex=ee_index,
            targetPosition=target_pos.tolist()
        )
        joint_positions = [joint_positions[j] for j in range(len(joint_indices))]
        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=joint_positions,
            forces=[200]*len(joint_indices)
        )
        step_sim(dt)

    pos_model = get_end_effector_pos(robot_id, ee_index)
    err_model = np.linalg.norm(pos_model - target_pos)
    print("Model-based final position:", pos_model, "error:", err_model)

    # --- Reset robot for data-driven demo ---
    time.sleep(1.0)
    for idx in joint_indices:
        p.resetJointState(robot_id, idx, 0.0)

    # --- 2) Data-driven approx ---
    print("Running data-driven IK approximation...")
    approx = DataDrivenIKApprox()

    for _ in range(steps):
        q_pred = approx.predict_joints(target_pos)
        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=q_pred.tolist(),
            forces=[200]*len(joint_indices)
        )
        step_sim(dt)

    pos_data = get_end_effector_pos(robot_id, ee_index)
    err_data = np.linalg.norm(pos_data - target_pos)
    print("Data-driven final position:", pos_data, "error:", err_data)

    print("\nComparison:")
    print("  Model-based error      =", err_model)
    print("  Data-driven error      =", err_data)
    print("You can visually compare in the GUI as well.")


# ==============================
#  MAIN
# ==============================

if __name__ == "__main__":
    # Run the three objectives one by one.
    # Close the GUI window if you want to re-run from scratch.
    demo_objective1_model_based_control()
    time.sleep(2.0)

    demo_objective2_nl_to_ltl()
    time.sleep(2.0)

    demo_objective3_model_vs_data_driven()

    print("\nAll demos finished. Close the PyBullet window to exit.")
