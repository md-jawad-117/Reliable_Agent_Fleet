"""Watch one cell live in the interactive MuJoCo viewer.

    python run_demo.py --mode baseline
    python run_demo.py --mode reliable
    python run_demo.py --mode reliable --no-faults
    python run_demo.py --mode baseline --speed 2      # 2x real time

Rotate/zoom with the mouse. The overlay shows live metrics; the event log
prints in this terminal.
"""
import argparse
import time

import mujoco
import mujoco.viewer

from fleet.net import demo_faults
from fleet.sim import CTRL_DT, Simulation


LEGEND = ("Lamp behind each arm:  green = working   blue = idle   red = crashed\n"
          "                       purple = cut off   yellow = backing off\n"
          "Airspace turns red on a collision.   Mouse: left-drag rotate, right-drag pan, scroll zoom")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "reliable"], default="reliable")
    ap.add_argument("--no-faults", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--ui", action="store_true", help="show MuJoCo's full side panels (for debugging)")
    args = ap.parse_args()

    sim = Simulation(args.mode, faults=[] if args.no_faults else demo_faults())
    printed = 0
    # clean view by default: no side panels, just the scene and a small overlay
    with mujoco.viewer.launch_passive(sim.model, sim.data,
                                      show_left_ui=args.ui, show_right_ui=args.ui) as v:
        v.cam.lookat[:] = (0.0, 0.05, 0.12)
        v.cam.distance, v.cam.elevation, v.cam.azimuth = 2.3, -50, 90
        v.opt.geomgroup[1] = 1
        wall0, sim0 = time.time(), sim.t
        while v.is_running():
            if not sim.done():
                with v.lock():
                    sim.step()
            lag = (sim.t - sim0) / args.speed - (time.time() - wall0)
            if lag > 0:
                time.sleep(lag)
            if int(sim.t / CTRL_DT) % 3 == 0 or sim.done():
                hud = sim.hud()
                v.set_texts([
                    (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                     "\n".join(hud), None),
                    (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                     LEGEND, None),
                ])
                v.sync()
            while printed < len(sim.log_lines):
                t, line = sim.log_lines[printed]
                if "start O" not in line:
                    print(f"{t:6.2f}  {line}")
                printed += 1
            if sim.done():
                time.sleep(0.02)
    print(sim.metrics().as_dict())


if __name__ == "__main__":
    main()
