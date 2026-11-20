"""Record the side-by-side demo video: baseline (left) vs reliable runtime (right).

Both cells get the exact same fault script. Output: media/demo_side_by_side.mp4

    python record_demo.py            # full video
    python record_demo.py --fast     # half resolution, for a quick look
"""
import argparse
import os
import textwrap

import cv2
import mujoco
import numpy as np

from fleet.net import demo_faults
from fleet.sim import Simulation

FRAME_EVERY = 3          # control ticks per video frame (0.03 s -> ~33 fps, real time)

GOOD = (90, 220, 120)
BAD = (80, 80, 255)
WHITE = (240, 240, 240)
GREY = (170, 170, 170)


def make_camera():
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0.0, 0.05, 0.12)
    cam.distance = 2.25
    cam.elevation = -52
    cam.azimuth = 90
    return cam


def put(img, text, org, color=WHITE, scale=0.5, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_hud(img, sim, s):
    m = sim.metrics()
    title = "BASELINE: retries + timeouts" if sim.mode == "baseline" else "RELIABLE RUNTIME: leases + fencing + reconcile"
    put(img, title, (int(12 * s), int(28 * s)), WHITE, 0.7 * s, 2)
    rows = [
        (f"t = {sim.t:5.1f} s", WHITE),
        (f"orders shipped correctly  {m.orders_ok}/{m.orders_total}", GOOD if m.orders_ok else WHITE),
        (f"duplicate shipments       {m.duplicate_shipments}", BAD if m.duplicate_shipments else GOOD),
        (f"arm-arm collisions        {m.collisions}", BAD if m.collisions else GOOD),
    ]
    for i, (txt, col) in enumerate(rows):
        put(img, txt, (int(12 * s), int((56 + 22 * i) * s)), col, 0.52 * s, 1)
    if sim.active_faults:
        put(img, "FAULT: " + ", ".join(sim.active_faults), (int(12 * s), int(150 * s)), BAD, 0.52 * s, 1)
    # lamp legend
    put(img, "lamp: green busy  blue idle  red crashed  purple cut off  yellow backing off",
        (int(12 * s), img.shape[0] - int(10 * s)), GREY, 0.4 * s, 1)


def draw_log(panel_w, h, sim, s):
    strip = np.full((h, panel_w, 3), 22, np.uint8)
    lines = [(t, l) for t, l in sim.log_lines if "start O" not in l][-6:]
    y = int(20 * s)
    for t, l in lines:
        col = BAD if ("COLLISION" in l or "FAULT" in l or "FAILED" in l or "again" in l) else (
            GOOD if ("camera" in l or "fenced" in l or "safely" in l or "re-validated" in l) else GREY)
        for j, part in enumerate(textwrap.wrap(f"{t:5.1f}s  {l}", 92)):
            put(strip, ("        " if j else "") + part, (int(10 * s), y), col, 0.42 * s, 1)
            y += int(17 * s)
    return strip


def final_card(w, h, sims):
    card = np.full((h, w, 3), 18, np.uint8)
    put(card, "Same robots, same faults, same planner", (40, 60), WHITE, 1.0, 2)
    put(card, "(lost acks, a controller crash, a network partition)", (40, 95), GREY, 0.7, 1)
    hdr = ["", "baseline", "reliable runtime"]
    rows = [("orders shipped correctly", "orders_ok", "/9"), ("duplicate shipments", "duplicate_shipments", ""),
            ("orders lost (out of stock)", "orders_failed", ""), ("arm-arm collisions", "collisions", ""),
            ("time to finish (s)", "makespan", "")]
    ms = [s.metrics().as_dict() for s in sims]
    x = [60, int(w * 0.55), int(w * 0.75)]
    for j, hname in enumerate(hdr):
        put(card, hname, (x[j], 170), WHITE, 0.9, 2)
    for i, (label, key, suf) in enumerate(rows):
        y = 220 + 50 * i
        put(card, label, (x[0], y), WHITE, 0.8, 1)
        for j, m in enumerate(ms):
            v = m[key]
            better = (v == max(ms[0][key], ms[1][key])) if key == "orders_ok" else (v == min(ms[0][key], ms[1][key]))
            col = WHITE if ms[0][key] == ms[1][key] or key == "makespan" else (GOOD if better else BAD)
            put(card, f"{v}{suf}", (x[j + 1], y), col, 0.9, 2)
    return card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", default=os.path.join("media", "demo_side_by_side.mp4"))
    args = ap.parse_args()
    s = 0.5 if args.fast else 1.0
    W, H, LOG_H = int(800 * s), int(600 * s), int(140 * s)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    sims = [Simulation("baseline", faults=demo_faults()), Simulation("reliable", faults=demo_faults())]
    renders = [mujoco.Renderer(sm.model, H, W) for sm in sims]
    cam = make_camera()
    opt = mujoco.MjvOption()
    opt.geomgroup[1] = 1
    fps = 1.0 / (FRAME_EVERY * 0.01)
    vw = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * W, H + LOG_H))
    frame = 0
    while not all(sm.done() for sm in sims):
        for sm in sims:
            if not sm.done():
                for _ in range(FRAME_EVERY):
                    sm.step()
        panels = []
        for sm, r in zip(sims, renders):
            r.update_scene(sm.data, cam, scene_option=opt)
            img = np.ascontiguousarray(r.render()[:, :, ::-1])
            draw_hud(img, sm, s)
            panels.append(np.vstack([img, draw_log(W, LOG_H, sm, s)]))
        out = np.hstack(panels)
        cv2.line(out, (W, 0), (W, H + LOG_H), (255, 255, 255), 2)
        vw.write(out)
        frame += 1
        if frame % 300 == 0:
            print(f"  t={sims[0].t:5.1f}s / {sims[1].t:5.1f}s")
    card = final_card(2 * W, H + LOG_H, sims)
    for _ in range(int(fps * 6)):
        vw.write(card)
    vw.release()
    cv2.imwrite(os.path.splitext(args.out)[0] + "_results.png", card)
    for sm in sims:
        print(sm.mode, sm.metrics().as_dict())
    print("wrote", args.out)


if __name__ == "__main__":
    main()
