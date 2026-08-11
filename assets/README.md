# assets

`so101_new_calib.nomesh.urdf` — the SO-101 kinematic chain, used only to verify
`lerobot_spot/leader_kinematics.py` in `tests/test_leader_kinematics.py`. Nothing
at runtime reads it: the chain is transcribed into that module so the package has
no URDF or placo dependency.

Source: [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100),
`Simulation/SO101/so101_new_calib.urdf`, Apache-2.0.

Modified only by stripping `<visual>` and `<collision>` nodes, so it loads
without the STL meshes. Joint origins, axes and limits are untouched — those are
the only parts kinematics depends on.

Note the `new_calib` / `old_calib` split upstream: this is the newer calibration.
If your leader predates it, re-verify against `so101_old_calib.urdf`.
