"""Low-level control for one Panda: damped-least-squares IK + smooth Cartesian moves.

This is the robot's "motor layer". It knows nothing about orders or networks;
the robot node (robot.py) tells it where to go.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .scene import HOME_Q

GRIP_OPEN = 255.0
GRIP_CLOSED = 0.0


def _min_jerk(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class ArmController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, idx: int, scratch: mujoco.MjData = None):
        self.m, self.d, self.idx = model, data, idx
        p = f"arm{idx}/"
        self.jnt = [model.joint(f"{p}joint{j}").id for j in range(1, 8)]
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.jnt])
        self.dadr = np.array([model.jnt_dofadr[j] for j in self.jnt])
        self.finger_qadr = [model.jnt_qposadr[model.joint(f"{p}finger_joint{k}").id] for k in (1, 2)]
        self.act = [model.actuator(f"{p}actuator{j}").id for j in range(1, 8)]
        self.grip_act = model.actuator(f"{p}actuator8").id
        self.site = model.site(f"{p}tcp").id
        self.lo = model.jnt_range[self.jnt, 0]
        self.hi = model.jnt_range[self.jnt, 1]
        self.scratch = scratch if scratch is not None else mujoco.MjData(model)
        self.q_des = HOME_Q.copy()
        self.grip = GRIP_OPEN
        self.target_rot = None
        self._seg = None          # (p0, p1, t0, duration)
        self.frozen = False

    # ---------- setup ----------
    def reset_home(self):
        self.d.qpos[self.qadr] = HOME_Q
        self.d.qpos[self.finger_qadr] = 0.04
        self.q_des = HOME_Q.copy()
        self.grip = GRIP_OPEN
        self._apply_ctrl()

    def teleport(self, target):
        """Place the arm directly at a Cartesian pose (used only at reset)."""
        q = self.q_des
        for _ in range(60):
            q = self.solve_ik(np.asarray(target, float), q, iters=1)
        self.q_des = q
        self.d.qpos[self.qadr] = q
        self._apply_ctrl()

    def capture_orientation(self):
        """Lock the tool orientation (pointing down) seen at the home pose."""
        self.target_rot = self.d.site_xmat[self.site].reshape(3, 3).copy()

    # ---------- kinematics ----------
    def tcp_pos(self) -> np.ndarray:
        return self.d.site_xpos[self.site].copy()

    def _fk_scratch(self, q):
        self.scratch.qpos[:] = self.d.qpos
        self.scratch.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.scratch)
        mujoco.mj_comPos(self.m, self.scratch)

    def solve_ik(self, target: np.ndarray, q0: np.ndarray, iters: int = 4) -> np.ndarray:
        q = q0.copy()
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        for _ in range(iters):
            self._fk_scratch(q)
            pos = self.scratch.site_xpos[self.site]
            rot = self.scratch.site_xmat[self.site].reshape(3, 3)
            e_pos = target - pos
            r_err = self.target_rot @ rot.T
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, r_err.flatten())
            e_rot = np.zeros(3)
            mujoco.mju_quat2Vel(e_rot, quat, 1.0)
            e = np.concatenate([e_pos, 0.5 * e_rot])
            mujoco.mj_jacSite(self.m, self.scratch, jacp, jacr, self.site)
            J = np.vstack([jacp[:, self.dadr], jacr[:, self.dadr]])
            lam = 0.05
            JJt = J @ J.T + lam**2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            # null-space pull toward a comfortable posture
            Jpinv = J.T @ np.linalg.solve(JJt, np.eye(6))
            dq += (np.eye(7) - Jpinv @ J) @ (0.05 * (HOME_Q - q))
            q = np.clip(q + dq, self.lo + 0.02, self.hi - 0.02)
        return q

    # ---------- motion ----------
    def move_to(self, target, t_now: float, speed: float = 0.45, min_dur: float = 0.5):
        target = np.asarray(target, dtype=float)
        start = self.commanded_pos()
        dur = max(min_dur, np.linalg.norm(target - start) / speed)
        self._seg = (start, target, t_now, dur)

    def commanded_pos(self) -> np.ndarray:
        self._fk_scratch(self.q_des)
        return self.scratch.site_xpos[self.site].copy()

    def done(self, t_now: float) -> bool:
        if self._seg is None:
            return True
        p0, p1, t0, dur = self._seg
        err = np.linalg.norm(self.tcp_pos() - p1)
        # normally within 2 cm; but if a joint limit stops the arm just short, accept 4 cm once the
        # move is well overdue, rather than waiting forever
        return t_now >= t0 + dur and (err < 0.02 or (t_now >= t0 + dur + 2.0 and err < 0.04))

    def hold_here(self):
        """Forget any in-progress motion and hold the current pose (what a rebooted controller does)."""
        self._seg = None
        self.q_des = self.d.qpos[self.qadr].copy()

    def set_grip(self, closed: bool):
        self.grip = GRIP_CLOSED if closed else GRIP_OPEN

    def update(self, t_now: float):
        """Called at the control rate. A frozen (crashed) arm just holds its last command."""
        if self.frozen:
            return
        if self._seg is not None:
            p0, p1, t0, dur = self._seg
            s = _min_jerk((t_now - t0) / dur)
            way = p0 + s * (p1 - p0)
            self.q_des = self.solve_ik(way, self.q_des)
        self._apply_ctrl()

    def _apply_ctrl(self):
        self.d.ctrl[self.act] = self.q_des
        self.d.ctrl[self.grip_act] = self.grip
