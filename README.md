# LeRobot-Spot

Teleoperate a Boston Dynamics Spot arm from a LeRobot SO-101 (or SO-100) leader
arm, in either **position** or **velocity** control — and collect imitation-learning
demonstrations by **hand-guiding** the arm directly.

```
SO-101 leader  ──►  retarget  ──►  Spot arm
 Feetech bus       joint map or      ArmJointMoveCommand   (position mode)
 6 servos          twist map         ArmVelocityCommand    (velocity mode)

your hand      ──►  admittance ──►  Spot arm
 on the gripper    deflection→v      ArmImpedanceCommand   (hand-guide mode)
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

## Step 0.5 — before the robot has ever moved

Two things are worth doing in this order, because together they remove the
biggest unknown at zero risk.

**Check the servos.** `probe_leader.py` is read-only and works before any
calibration exists — it pings all six servos and streams raw encoder counts:

```bash
python scripts/probe_leader.py --port /dev/tty.usbmodem59700725491
```

**Establish the sign conventions without commanding anything.** Connect to the
real robot in `--dry-run`, which reads live state and computes targets but never
sends an arm command. Drive the arm *from the tablet* and watch the `spot (deg)`
row to learn which way each Spot joint counts. Then move the leader and watch the
`target` row. Comparing the two settles every sign with the motors never taking
an order. Fix mismatches by flipping `sign` in a config.

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

## First contact with a powered arm

Spot sitting, arm unstowed, open space, **tablet in someone's hand with the
E-Stop under their thumb**. Leave `--estop` at its default so that button keeps
working. Take one joint at a time:

```bash
python -m lerobot_spot.teleop $SPOT_IP --leader-port /dev/tty.usbmodem59700725491 \
    --leader-id spot_leader --config configs/first_contact.json --only-joint sh0
```

`--only-joint` freezes every other joint, so a sign error can only move the one
you are watching. `configs/first_contact.json` uses gain 0.2 (90° of leader
travel gives 18° of Spot), `max_joint_vel` 0.35 rad/s and a 3° deadband, so a
mistake is a slow drift rather than a swing. Walk `sh0 → sh1 → el0 → wr0 → wr1`,
then drop `--only-joint` and raise the gain with `]` until it feels right.

Velocity mode is a different command path, so re-earn trust from this step before
trusting it.

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

### Who owns the E-Stop

`--estop leave` is the default, and it matters more than anything else on this
page. Boston Dynamics' examples call `EstopEndpoint.force_simple_setup()`, which
*replaces the robot's entire E-Stop configuration with a single endpoint* -- the
script's own. That unregisters the tablet, so **the tablet's red button stops
working** for as long as the script runs. A tool that has never driven a real arm
should not be the only thing that can stop it.

With `leave` we register no endpoint at all: the tablet keeps E-Stop authority
and someone else must release the E-Stop before the motors will power. `SPACE`
does nothing but say so. The status line shows `(tablet)` or `(ours)` so there is
never any doubt, and the released/asserted state is read from the robot rather
than from our own keep-alive, so it stays truthful either way.

Use `--estop take` only when nobody is holding a tablet. Then `SPACE` is your
only software stop and the endpoint's 9-second timeout is your only backstop.

## Hand-guided demonstration collection

```bash
python -m lerobot_spot.collect ROBOT_IP --output ~/demos/pick-cup --task "pick up the cup"
```

Grab the gripper, drag it through the task in all six degrees of freedom, and
press `R` to bracket each take. This is the Franka Panda kinesthetic-teaching
workflow, reached a different way.

### Why it is not the same thing a Panda does

A Panda has joint torque sensors and a backdrivable transmission, so it can null
out its own gravity and friction and go limp under your hand. **Spot cannot.**
Its arm is not backdrivable and the SDK exposes no joint torque interface, so
"go limp" is not a command that exists on this robot.

What Spot does expose is `ArmImpedanceCommand`: a virtual 6-DOF spring-damper
between a *desired tool* frame you stream and the *tool* frame the arm reaches.
That alone is not hand-guiding — push the arm and it springs straight back,
because the setpoint never moved. So `handguide.py` closes an outer **admittance**
loop around it:

```
you push  →  tool deflects from the setpoint
          →  deflection read back from impedance feedback
          →  deflection drives the setpoint in the push direction
          →  arm follows the setpoint, i.e. follows your hand
```

Let go and the deflection decays to zero, so the setpoint stops on its own. That
self-termination is the property that makes the loop safe: nothing keeps
integrating once you stop pushing.

Practically, expect it to feel like dragging something through honey rather than
moving a weightless Panda. Friction in Spot's harmonic drives is real and this
loop does not cancel it — it works around it. Below `--force-deadband` newtons of
push, nothing moves at all.

### Where the push signal comes from

`--wrench-source deflection` (default) infers your push from
`desired_tool_tform_tool` times the stiffness matrix — the same quantity the
robot computes internally, and exact forward kinematics with nothing estimated.

`--wrench-source measured` uses Spot's own wrench estimate, derived from joint
currents. Truer to real contact force, but biased by payload and friction, so it
needs `b` and will still drift as the arm's pose changes. It is also the mode in
which the leash actually matters — see below.

### Keys

| Key | Action |
| --- | --- |
| `e` | engage / disengage hand-guiding |
| `R` | start / stop a take |
| `D` | discard the take in progress |
| `f` | freeze the setpoint — let go and reposition without drift |
| `b` | zero the loop against a payload (let go of the arm first) |
| `o` / `p` | gripper open / close |
| `[` `]` | scale the admittance gains live |
| `ESC` | stop, disengage, and save the take |

The stand/sit/power/lease/E-Stop keys are the same as teleop.

### Carrying something

A payload in the gripper pulls down forever, which the loop reads as a permanent
downward push and acts on — the arm sinks. Let go of the arm and press `b`.

What that does depends on the wrench source, because the two need opposite
corrections. In `deflection` mode the payload shows up as a standing sag, so its
weight is handed to Spot as a feed-forward wrench and the sag itself goes away.
In `measured` mode a feed-forward would not help — the arm still reports the
force it exerts to hold the load — so the resting wrench is subtracted as a bias
instead. Applying both would double-count the load and drift the arm upward.

### Safety

Read the teleop safety section first; all of it still applies. What is different
here is that **you are inside the workspace with your hands on the arm**, which
teleop never asks of you. On top of the usual watchdogs:

- **The leash.** The setpoint is never allowed further than `--max-deflection`
  from the tool the arm actually reached, which bounds the spring force at
  `linear_stiffness × max_deflection` — 15 N at the defaults. This is the single
  most important number in the config.
- **`max_force_mag` / `max_torque_mag`** are sent with every command, so the
  robot saturates its own output at 30 N / 8 Nm, below the API's 60 N / 15 Nm
  defaults. Independent of the leash, on purpose.
- **The workspace box** clamps the setpoint into a body-relative box and a reach
  annulus, so it cannot be dragged past where the arm can follow. The defaults
  are conservative guesses — **tune them for your workspace before trusting them.**
- **Instability detection.** If the arm reports `STATUS_TRAJECTORY_CANCELLED` it
  has detected its own instability; the loop disengages and tells you to lower
  the stiffness. Re-engaging at the same stiffness will just repeat it.
- Engaging seeds the setpoint at the arm's current pose, so it can never jump —
  it is safe to engage with your hands already on the gripper.

Disengaging does *not* cut the command. It leaves the spring holding the arm
where you left it, which is a still, compliant, still-powered state — not a
limp one. `ESC` is the real stop.

Bring the arm up on `--dry-run` first and watch the push and deflection readouts
respond to your hand before you let it drive anything.

### Tuning

| Flag / key | Default | Effect |
| --- | --- | --- |
| `--linear-stiffness` | `150` N/m | lower is easier to push, and sags more |
| `--angular-stiffness` | `12` Nm/rad | as above, for rotation |
| `--linear-damping` | `3.0` Ns/m | raise if the arm feels bouncy |
| `--force-deadband` | `3.0` N | push needed before anything moves |
| `--linear-admittance` | `0.010` (m/s)/N | how fast a given push drags the arm |
| `--linear-speed-limit` | `0.15` m/s | ceiling on setpoint speed |
| `--max-deflection` | `0.10` m | the leash |
| `velocity_cutoff_hz` | `2.0` | the "virtual inertia" knob; lower is heavier |
| `--tool-offset` | `0.196 0 0` | put this where your hand actually grips |

Stiffness and damping are clamped into the envelope Boston Dynamics' own
impedance example runs in (≤500 N/m, ≤60 Nm/rad); above that the arm is
documented to go unstable. If it oscillates, lower stiffness first, then lower
both. If it feels sluggish, raise `--linear-admittance` with `]` rather than
touching the spring.

`--tool-offset` matters more than it looks: it is the point the virtual spring
pulls on. Put it where your hand actually is and rotations will feel natural;
leave it wrong and the arm will fight you when you twist.

### What gets recorded

One directory per take, holding a `samples.jsonl` written tick by tick (the
crash-proof copy) and a `samples.npz` written at the end (the trainable one).
`recorder.load_episode()` reads either back as a dict of `(ticks, …)` arrays.

Per tick: joint position / velocity / load (6 each, in `SPOT_JOINTS` order);
hand pose in odom, vision and body (7 each, `xyz` + `wxyz`); hand twist in odom
and vision (6 each); body pose in odom and vision; Spot's estimated end-effector
force and wrench; gripper opening, command and holding flag; the impedance
setpoint, the deflection, the inferred operator wrench, the commanded setpoint
twist, the commanded and measured tool wrenches, the live stiffness/damping/
feed-forward, and the impedance status plus the engaged / frozen / leashed /
clamped flags.

Discarded takes are deleted outright and their index is reused, so a session
directory holds only takes you kept.

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

No robot and no leader arm required. 153 pass with only the stub installed; 150
pass against the real Spot SDK (the difference is tests that assert on the
stub's own command shapes, which skip when the real SDK is present).

The retargeting and hand-guiding tests are pure maths — `test_handguide.py` pins
the safety properties directly (engaging never moves the arm, releasing stops the
setpoint, the leash bounds the spring force). The teleop and collect tests drive
their state machines: engage gating, the watchdogs, the home-anchor alignment
gate, and the commands that reach the wire.

`test_sdk_contract.py` is the important one. Every other test can pass while the
code is wrong, because `tests/fake_bosdyn/` encodes the same beliefs about the
Spot API that the production code does — if a belief is mistaken, the stub is
mistaken identically and the suite stays green. The contract tests break that
circularity: they are skipped unless the genuine `bosdyn` package is installed,
and they inspect the real protobuf messages the builders emit. They still need no
robot and open no connection.

They check that joint order reaches the proto unpermuted, that `max_vel`/`max_acc`
are really applied, that folding the gripper in does not drop the arm sub-command
in either control mode, that `open_fraction` 0 is closed and 1 is open, that a
zero twist is genuinely all-zero, and that our joint limits never exceed Boston
Dynamics' published URDF. That last check earned its keep: it caught five limits
that had been rounded *outward* past the hard stops.

**Re-run this against the SDK version your robot actually runs, and after every
SDK upgrade.** A failure means a command would have been malformed on the wire.

## Layout

```
lerobot_spot/
  leader.py     threaded reader for the SO-101, tolerant of LeRobot's module moves
  retarget.py   joint map, twist map, filters, limits, the two anchors
  handguide.py  admittance law, SE(3) helpers, the leash and workspace clamps
  recorder.py   episode writer (streaming jsonl + trainable npz) and loader
  spot_arm.py   lease / E-Stop / power / state, and the three command paths
  teleop.py     leader-driven control loop, curses UI, CLI
  collect.py    hand-guided control loop, curses UI, CLI
scripts/
  probe_leader.py   pings the servos, streams raw counts, needs no calibration
  check_leader.py   leader-only sanity check, no robot
configs/
  so101_to_spot.example.json
  first_contact.json          timid profile for the first powered run
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
- Hand-guiding does not cancel the arm's friction, only work around it. Fine
  positioning below roughly the `--force-deadband` threshold is not achievable by
  hand; use teleop for that.
- The hand-guide loop spends two RPCs per tick (command + feedback). If the
  measured rate in the header falls well below `--rate`, lower `--rate` — the
  admittance law is written to be correct at any rate, but a laggy loop feels
  worse than a slower one.
- No camera frames are recorded. The episodes hold proprioception and wrench
  only; pair them with your own image capture if the policy needs pixels.
- The workspace box and reach annulus defaults are guesses at Spot's geometry,
  not surveyed values. Tune them before relying on them.
