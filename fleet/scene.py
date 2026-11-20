"""Builds the MuJoCo work cell: three real Franka Panda arms around a shared table.

Layout (top view):
  - A shared table in the middle holds a 3x3 grid of coloured boxes (the "stock").
  - Three Pandas sit around the table, 120 degrees apart, all facing the centre.
  - Each arm has its own shipping bin on the floor beside it.
  - The "airspace" above the table is a shared physical resource: two arms
    reaching into it at the same time can collide.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANDA_DIR = os.path.join(ROOT, "third_party", "mujoco_menagerie", "franka_emika_panda")

N_ARMS = 3
ARM_RADIUS = 0.56          # distance of each arm base from the table centre
TABLE_TOP = 0.10           # table surface height (m)
TABLE_HALF = 0.24
BOX_HALF = 0.025           # 5 cm cubes
GRID_SPACING = 0.12
BIN_RADIUS = 0.80
BIN_ANGLE_OFFSET = math.radians(42)
BIN_HALF = 0.09
AIRSPACE_RADIUS = 0.26     # cylinder above the table that only one arm may enter

HOME_Q = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 1.9, -0.7853])

COLORS = {
    "red": (0.85, 0.15, 0.15, 1),
    "green": (0.15, 0.7, 0.25, 1),
    "blue": (0.15, 0.35, 0.9, 1),
}
# 3x3 stock grid, colour per cell (row-major)
GRID_COLORS = ["red", "green", "blue",
               "blue", "red", "green",
               "green", "blue", "red"]


@dataclass
class CellLayout:
    arm_base: list = field(default_factory=list)     # (x, y) per arm
    arm_yaw: list = field(default_factory=list)
    bin_pos: list = field(default_factory=list)      # (x, y) per arm
    staging: list = field(default_factory=list)      # (x, y, z) hover point outside airspace
    box_start: list = field(default_factory=list)    # (x, y, z) per box
    box_color: list = field(default_factory=list)


def compute_layout() -> CellLayout:
    lay = CellLayout()
    for i in range(N_ARMS):
        theta = math.radians(90 + 120 * i)
        lay.arm_base.append((ARM_RADIUS * math.cos(theta), ARM_RADIUS * math.sin(theta)))
        lay.arm_yaw.append(theta + math.pi)  # face the centre
        b = theta + BIN_ANGLE_OFFSET
        lay.bin_pos.append((BIN_RADIUS * math.cos(b), BIN_RADIUS * math.sin(b)))
        lay.staging.append((0.36 * math.cos(theta), 0.36 * math.sin(theta), 0.42))
    k = 0
    for r in range(3):
        for c in range(3):
            x = (c - 1) * GRID_SPACING
            y = (1 - r) * GRID_SPACING
            lay.box_start.append((x, y, TABLE_TOP + BOX_HALF))
            lay.box_color.append(GRID_COLORS[k])
            k += 1
    return lay


def _world_xml(lay: CellLayout) -> str:
    parts = []
    # status lamps, bins, boxes
    for i, (bx, by) in enumerate(lay.bin_pos):
        h, w, t = BIN_HALF, 0.05, 0.006
        parts.append(f"""
    <body name="bin{i}" pos="{bx} {by} 0">
      <geom type="box" size="{h} {h} 0.004" pos="0 0 0.004" rgba="0.35 0.3 0.25 1"/>
      <geom type="box" size="{h} {t} {w}" pos="0 {h} {w}" rgba="0.45 0.38 0.3 1"/>
      <geom type="box" size="{h} {t} {w}" pos="0 {-h} {w}" rgba="0.45 0.38 0.3 1"/>
      <geom type="box" size="{t} {h} {w}" pos="{h} 0 {w}" rgba="0.45 0.38 0.3 1"/>
      <geom type="box" size="{t} {h} {w}" pos="{-h} 0 {w}" rgba="0.45 0.38 0.3 1"/>
    </body>""")
    for i, (ax, ay) in enumerate(lay.arm_base):
        # pedestal lamp behind the arm shows its health
        th = math.atan2(ay, ax)
        lx, ly = ax + 0.16 * math.cos(th), ay + 0.16 * math.sin(th)
        parts.append(f"""
    <geom name="lamp{i}" type="cylinder" size="0.035 0.12" pos="{lx} {ly} 0.12"
          rgba="0.3 0.3 0.3 1" contype="0" conaffinity="0"/>""")
    for k, ((x, y, z), col) in enumerate(zip(lay.box_start, lay.box_color)):
        r, g, b, a = COLORS[col]
        parts.append(f"""
    <body name="box{k}" pos="{x} {y} {z}">
      <freejoint name="box{k}_free"/>
      <geom name="box{k}_geom" type="box" size="{BOX_HALF} {BOX_HALF} {BOX_HALF}"
            rgba="{r} {g} {b} {a}" mass="0.08" friction="1.2 0.02 0.001"/>
    </body>""")
    body = "".join(parts)
    return f"""
<mujoco model="reliable_agent_fleet">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <size memory="24M"/>
  <visual>
    <headlight diffuse="0.35 0.35 0.35" ambient="0.22 0.22 0.24" specular="0 0 0"/>
    <global azimuth="120" elevation="-35" offwidth="1280" offheight="720"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.80 0.83 0.88" rgb2="0.45 0.49 0.56" width="512" height="512"/>
    <texture type="2d" name="floor" builtin="checker" mark="edge" rgb1="0.62 0.64 0.67" rgb2="0.58 0.60 0.63"
             markrgb="0.50 0.52 0.56" width="300" height="300"/>
    <material name="floor" texture="floor" texuniform="true" texrepeat="6 6" reflectance="0"/>
    <material name="table" rgba="0.62 0.55 0.47 1"/>
  </asset>
  <worldbody>
    <light pos="0.5 -0.8 3" dir="-0.15 0.25 -1" directional="true" castshadow="true" diffuse="0.5 0.5 0.5"/>
    <light pos="-1 1 2.5" dir="0.3 -0.3 -1" directional="true" castshadow="false" diffuse="0.2 0.2 0.22"/>
    <geom name="floor" type="plane" size="3 3 0.05" material="floor"/>
    <geom name="table" type="box" size="{TABLE_HALF} {TABLE_HALF} {TABLE_TOP / 2}" pos="0 0 {TABLE_TOP / 2}"
          material="table"/>
    <geom name="airspace" type="cylinder" size="{AIRSPACE_RADIUS} 0.22" pos="0 0 {TABLE_TOP + 0.22}"
          rgba="0.2 0.8 1.0 0.06" contype="0" conaffinity="0" group="1"/>
    {body}
  </worldbody>
</mujoco>"""


def _yaw_quat(yaw: float) -> list:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def build_model() -> tuple[mujoco.MjModel, CellLayout]:
    lay = compute_layout()
    spec = mujoco.MjSpec.from_string(_world_xml(lay))
    for i in range(N_ARMS):
        panda = mujoco.MjSpec.from_file(os.path.join(PANDA_DIR, "panda.xml"))
        panda.meshdir = os.path.join(PANDA_DIR, "assets")
        for key in list(panda.keys):
            panda.delete(key)
        # tool-centre point between the fingertips
        panda.body("hand").add_site(name="tcp", pos=[0, 0, 0.1034], size=[0.01, 0, 0], rgba=[1, 1, 0, 0.0])
        ax, ay = lay.arm_base[i]
        frame = spec.worldbody.add_frame(pos=[ax, ay, 0.0], quat=_yaw_quat(lay.arm_yaw[i]))
        spec.attach(panda, prefix=f"arm{i}/", frame=frame)
    model = spec.compile()
    return model, lay
