# bridge.py — bascule MOCK↔︎RÉEL plus tard
import time, numpy as np, json
from avp_stream import VisionProStreamer

USE_MOCK = True  # passe à False quand tu branches le vrai bus

if USE_MOCK:
    class Bus:
        def set_goal_positions(self, d): print("[MOCK] set_goal_positions", d)
else:
    from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
    from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
    cfg = FeetechMotorsBusConfig(
        port="/dev/tty.usbmodemXXXX",
        motors={"j1":(1,"sts3215"),"j2":(2,"sts3215"),"j3":(3,"sts3215"),
                "j4":(4,"sts3215"),"j5":(5,"sts3215"),"j6":(6,"sts3215")}
    )
    Bus = FeetechMotorsBus(cfg)

bus = Bus()
s = VisionProStreamer(ip="IP_DU_CASQUE", record=False)

# paramètres sécurités/mapping
JLIMITS = {1:(-1.2,1.2), 2:(-0.8,1.2), 3:(-1.2,1.0), 4:(-1.2,1.2), 5:(-1.5,1.5), 6:(0,0.8)}
MAX_DQ  = 2.0  # rad/s (limite vitesse)
ALPHA   = 0.2  # low-pass

q_prev   = {i:0.0 for i in range(1,7)}
t_prev   = time.time()
last_rx  = time.time()

def clamp(v, lo, hi): return float(max(lo, min(hi, v)))

while True:
    r = s.latest
    now = time.time()
    if (now - last_rx) > 0.3:  # dead-man: plus de flux → stop
        bus.set_goal_positions({i: q_prev[i] for i in range(1,6)})
        continue
    last_rx = now

    T = r["right_wrist"][0]       # 4x4
    yaw = float(np.arctan2(T[1,0], T[0,0]))
    x,y,z = T[0,3], T[1,3], T[2,3]

    # mapping simple (à raffiner plus tard / IK)
    goals = {
        1: clamp( 2.0*x, *JLIMITS[1]),
        2: clamp( 0.5+2.0*y, *JLIMITS[2]),
        3: clamp(-0.3+2.5*z, *JLIMITS[3]),
        4: 0.0,
        5: clamp(yaw, *JLIMITS[5]),
    }
    # lissage + limite de vitesse
    dt = max(1e-3, now - t_prev)
    for j in goals:
        q_target = goals[j]
        q_lp = ALPHA*q_target + (1-ALPHA)*q_prev[j]
        dq = clamp((q_lp - q_prev[j])/dt, -MAX_DQ, MAX_DQ)
        goals[j] = q_prev[j] + dq*dt
    t_prev = now; q_prev.update(goals)

    bus.set_goal_positions(goals)
