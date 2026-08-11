# LeRobot-Spot

Teleoperate a Boston Dynamics Spot arm from a LeRobot SO-101 (or SO-100) leader
arm, in either **position** or **velocity** control.

```
SO-101 leader  ──►  retarget  ──►  Spot arm
 Feetech bus       joint map or      ArmJointMoveCommand   (position mode)
 6 servos          twist map         ArmVelocityCommand    (velocity mode)
```

## Why joint-space works here

The SO-101 is very nearly a kinematic subset of Spot's arm. Every leader joint
has a Spot joint with the same axis:

| SO-101 joint    | axis       | Spot joint | axis       | Spot range (rad) |
| --------------- | ---------- | ---------- | ---------- | ---------------- |
| `shoulder_pan`  | yaw (z)    | `sh0`      | yaw (z)    | −2.618 … 3.142   |
| `shoulder_lift` | pitch (y)  | `sh1`      | pitch (y)  | −3.142 … 0.524   |
| `elbow_flex`    | pitch (y)  | `el0`      | pitch (y)  | 0 … 3.142        |
| —               | —          | `el1`      | roll (x)   | −2.793 … 2.793   |
| `wrist_flex`    | pitch (y)  | `wr0`      | pitch (y)  | −1.833 … 1.833   |
| `wrist_roll`    | roll (x)   | `wr1`      | roll (x)   | −2.880 … 2.880   |
| `gripper`       | trigger    | claw       | open frac. | 0 … 1            |

So no IK is needed: the map is one joint to one joint. The only gap is Spot's
forearm roll `el1`, which the leader has no counterpart for — it is held wherever
it was when you engaged. (Limits are from Boston Dynamics' published arm model.)

## Install

```bash
pip install -r requirements.txt
```

Then calibrate the leader once with LeRobot's own tooling, or let the first run
walk you through it (`LeaderArm.connect()` calls LeRobot's `calibrate()` when no
calibration file matches the `--leader-id`).

## Step 0 — check the leader on its own

No robot involved. Do this every time you change the mapping or the wiring.

```bash
python scripts/check_leader.py --leader-port /dev/tty.usbmodem585A0077181 --leader-id my_leader
```

Move one joint at a time and confirm the Spot column moves the way you want. **A
joint that runs backwards here will run backwards on the robot.** Fix it by
copying `configs/so101_to_spot.example.json`, flipping that joint's `sign`, and
passing `--config`.

## Step 1 — run it

```bash
python -m lerobot_spot.teleop $SPOT_IP \
    --leader-port /dev/tty.usbmodem585A0077181 \
    --leader-id my_leader \
    --mode position \
    --dry-run
```

`--dry-run` connects, reads Spot's real joint state and computes targets, but
never sends an arm command. The `target` row in the UI shows exactly what would
have been commanded. Watch it for a minute before dropping the flag.

Credentials come from the Spot SDK's usual `BOSDYN_CLIENT_USERNAME` /
`BOSDYN_CLIENT_PASSWORD` environment variables, or it prompts.

## Startup: both arms begin stowed

The leader arm and Spot's arm both start folded. That is worth exploiting rather
than working around, because stow is the one pose each arm returns to *exactly* —
which makes it a free, repeatable correspondence between the two.

Recommended first run:

1. Leave **both arms stowed**. Start the script (with `--anchor home`).
2. Press **`SPACE`** to release the software E-Stop, **`P`** to power on,
   **`c`** to stand.
3. Press **`H`**. This captures the stowed leader pose together with Spot's
   stowed joint angles as the home correspondence. Add `--save-home home.json`
   to write it out so later sessions can load it with `--config`.
4. Press **`y`** to unstow Spot's arm.
5. Unfold the leader. The `align` row shows, per joint, how far the two are out
   of correspondence. Bring it near zero.
6. Press **`e`** to engage. It refuses if any joint is more than
   `home_tolerance_deg` (default 15°) out; press `e` again within 3 s to force.

At the end of a session: press **`e`** to disengage, **`h`** to stow the arm,
**`v`** to sit, **`P`** to power off. Fold the leader back to stow so the next
run's `H` capture is consistent.

### The two anchors

| | `--anchor current` (default) | `--anchor home` |
| --- | --- | --- |
| Reference | wherever both arms are at engage | one captured pose pair |
| Jump on engage | impossible, delta is zero | gated, and rate limited even if forced |
| Leader pose means | something different after each re-index | a fixed Spot pose all session |
| Re-indexing | yes, `r` or disengage/re-engage | breaks correspondence until re-engage |

`current` is the safer default and works with no setup. `home` is what you want
when you care that a given leader pose always means the same Spot pose — for
demonstrations you intend to record and replay, for instance.

Velocity mode is inherently relative (the leader is a joystick centred wherever
you engaged), so the anchor setting does not apply to it.

## Control modes

**Position** (`--mode position`) streams `ArmJointMoveCommand` trajectories, one
knot point `--lookahead` seconds ahead, replaced every tick. The arm mirrors the
leader's posture. Each target is EMA-smoothed, rate limited to
`--max-joint-vel`, and clamped inside the joint limits with a margin.

**Velocity** (`--mode velocity`) streams `ArmVelocityCommand`. Leader
displacement from the engage pose becomes hand velocity — the leader acts as a
spring-less 5-axis joystick. Good for reaching further than the leader's own
workspace allows.

Default velocity bindings (all rebindable in the config):

| Leader joint | Twist axis | Meaning |
| --- | --- | --- |
| `shoulder_pan` | `v_theta` | swing hand around the body |
| `shoulder_lift` | `v_z` | raise / lower hand |
| `elbow_flex` | `v_r` | extend / retract hand |
| `wrist_flex` | `v_ry` | hand pitch |
| `wrist_roll` | `v_rx` | hand roll |
| — | `v_rz` | hand yaw, unmapped by default |

Five leader joints cannot cover six twist axes, so `v_rz` is left out; `v_theta`
already swings the hand around the body. Rebind in `twist_map` if you would
rather trade one of the others away.

Switch modes live with `m` (it disengages first).

## Keys

| Key | Action |
| --- | --- |
| `e` | engage / disengage the clutch |
| `r` | re-index at the current leader pose |
| `H` | capture the home correspondence (do this stowed) |
| `m` | switch position ↔ velocity |
| `g` | gripper passthrough on / off |
| `[` `]` | scale the gain / velocity scale down / up |
| `SPACE` | toggle the software E-Stop |
| `P` | toggle motor power |
| `x` | release / re-acquire the lease |
| `c` / `v` | stand / sit |
| `y` / `h` | unstow / stow the arm |
| `ESC` | stop the robot and disengage |
| `TAB` | disengage and quit |

## Safety

Nothing moves unless you are engaged, and engaging is an explicit act. On top of
that, the loop disengages by itself when:

- the leader reading goes stale (serial unplugged, thread wedged) — >250 ms old;
- any leader joint moves more than `--max-leader-jump` degrees in one tick
  (default 15°), which is what a dropped or knocked leader looks like;
- the lease is lost, or the motors power off;
- the control step raises — it disengages *and* sends a stop.

In velocity mode, disengaging sends an explicit zero twist rather than waiting
for the command to expire. Velocity commands also carry a 0.5 s expiry, so if
this process dies outright the arm coasts to a stop within that window instead of
continuing at its last speed.

`--clutch hold` turns `e` into a dead-man that needs terminal key-repeat to hold
it down; it disengages `--hold-timeout` (default 0.7 s) after the last keypress.
The default `--clutch toggle` is less elegant but disengages the instant you
press the key, which is why it is the default. Whichever you pick, `ESC` is the
real stop button.

**The software E-Stop is not a substitute for the hardware E-Stop.** Keep the
tablet or a hardware estop within reach.

## Tuning

Everything below lives in a JSON config (`--config`); see
`configs/so101_to_spot.example.json`. Anything you omit keeps its default.

| Key | Default | Effect |
| --- | --- | --- |
| `joint_map[j].sign` | `1.0` | flip a joint's direction |
| `joint_map[j].gain` | `1.0` | leader degrees → Spot degrees |
| `deadband_deg` | `1.0` | ignore leader motion below this |
| `ema_alpha` | `0.35` | lower is smoother and laggier; `1.0` disables |
| `max_joint_vel` | `1.5` | rad/s ceiling in position mode |
| `limit_margin_rad` | `0.05` | stay this far off the hard stops |
| `home_tolerance_deg` | `15.0` | how aligned you must be to engage on `home` |
| `linear_scale` | `0.5` | velocity mode, normalized |
| `angular_scale` | `1.0` | velocity mode, rad/s |
| `twist_map[j].span_deg` | `45.0` | leader travel that saturates an axis |

`[` and `]` scale gains live, so you can find the right feel while running and
then write the number into the config.

## Tests

```bash
python -m pytest tests/ -q
```

71 tests, no robot and no leader arm required. The retargeting tests are pure
maths; the teleop tests drive the state machine — engage gating, the watchdogs,
the home-anchor alignment gate, and the shape of the commands that reach the wire
— against `tests/fake_bosdyn/`, a stub used only when the real SDK is absent.

## Layout

```
lerobot_spot/
  leader.py     threaded reader for the SO-101, tolerant of LeRobot's module moves
  retarget.py   joint map, twist map, filters, limits, the two anchors
  spot_arm.py   lease / E-Stop / power / state, and the two command paths
  teleop.py     control loop, curses UI, CLI
scripts/
  check_leader.py   leader-only sanity check, no robot
configs/
  so101_to_spot.example.json
```

## Known gaps

- Spot's forearm roll (`el1`) is never driven; the SO-101 has no joint for it.
  Reaching it needs either a 6-DOF leader or a modifier binding.
- Position mode has no Cartesian path guarantee. Joint-space interpolation
  between two poses can swing the elbow through places you did not intend —
  move deliberately near obstacles, or use velocity mode, which is Cartesian.
- No collision checking against Spot's own body. The joint clamps only keep each
  joint inside its own range.
- `--clutch hold` depends on terminal key-repeat, which macOS can have disabled.
  Verify it engages and releases as you expect before relying on it.
