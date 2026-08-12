"""Backport of the gymnasium-robotics 1.4 reset fix to the 1.3.x v3 Fetch envs.

In gymnasium-robotics 1.3.x, ``MujocoFetchEnv._reset_sim`` only calls
``mj_resetData``, which restores the *XML default* state — the arm's home
pose, with the gripper ~0.55 m away from the position that ``_env_setup``
moved it to at construction time and that ``_sample_goal`` samples goals
around. The captured ``initial_time/qpos/qvel`` are never restored, so every
episode starts 0.43–0.67 m from the goal instead of the documented <= 0.26 m.

Upstream fixed this in the v4 environments (gymnasium-robotics 1.4.0) by
restoring the captured initial state right after ``mj_resetData``. This module
monkeypatches the identical fix into the 1.3.x class so that FetchReach-v3
behaves as documented. It changes no physics, reward, or task logic and is
applied equally to the sparse and dense variants.
"""

import numpy as np
from gymnasium_robotics.envs.fetch.fetch_env import MujocoFetchEnv


def _reset_sim_v4(self):
    # Reset buffers for joint states, actuators, warm-start, control buffers etc.
    self._mujoco.mj_resetData(self.model, self.data)

    self.data.time = self.initial_time
    self.data.qpos[:] = np.copy(self.initial_qpos)
    self.data.qvel[:] = np.copy(self.initial_qvel)
    if self.model.na != 0:
        self.data.act[:] = None

    # Randomize start position of object.
    if self.has_object:
        object_xpos = self.initial_gripper_xpos[:2]
        while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
            object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                -self.obj_range, self.obj_range, size=2
            )
        object_qpos = self._utils.get_joint_qpos(
            self.model, self.data, "object0:joint"
        )
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        self._utils.set_joint_qpos(self.model, self.data, "object0:joint", object_qpos)

    self._mujoco.mj_forward(self.model, self.data)
    return True


def apply_reset_fix():
    MujocoFetchEnv._reset_sim = _reset_sim_v4
