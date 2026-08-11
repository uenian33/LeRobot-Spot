#!/usr/bin/env python3
"""
Replay recorded demonstration trajectories on Boston Dynamics Spot.

Usage:
    python replay_demonstration.py --hostname SPOT_IP --demo-path manual_traj_records/demo_XXXXXX/
    
Options:
    --speed-factor: Playback speed multiplier (default: 1.0, use 0.5 for half speed)
    --loop: Loop the demonstration continuously
    --skip-stand: Skip the stand command (if robot is already standing)
"""

import argparse
import os
import sys
import time
import numpy as np

# Boston Dynamics imports
import bosdyn.client
import bosdyn.client.util
from bosdyn.client import create_standard_sdk
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, get_a_tform_b
from bosdyn.client import math_helpers
from bosdyn.api import geometry_pb2


def load_demonstration(demo_path):
    """Load demonstration data from directory."""
    arm_poses_file = os.path.join(demo_path, "arm_poses.npy")
    
    if not os.path.exists(arm_poses_file):
        raise FileNotFoundError(f"arm_poses.npy not found in {demo_path}")
    
    arm_poses = np.load(arm_poses_file)
    print(f"Loaded {len(arm_poses)} poses from {arm_poses_file}")
    
    # Load metadata if available
    metadata_file = os.path.join(demo_path, "metadata.npy")
    if os.path.exists(metadata_file):
        metadata = np.load(metadata_file, allow_pickle=True).item()
        print(f"  Duration: {metadata.get('duration', 'unknown'):.2f}s")
        print(f"  Format: {metadata.get('arm_pose_format', 'unknown')}")
    
    return arm_poses


def replay_trajectory(robot, command_client, robot_state_client, arm_poses, 
                      speed_factor=1.0, loop=False):
    """
    Replay the recorded trajectory on the robot.
    
    Args:
        robot: Robot object
        command_client: RobotCommandClient
        robot_state_client: RobotStateClient  
        arm_poses: numpy array of shape (N, 9) with format:
                   [timestamp, x, y, z, qx, qy, qz, qw, gripper_pct]
        speed_factor: Playback speed multiplier (1.0 = original speed)
        loop: Whether to loop the trajectory
    """
    print("\n" + "="*60)
    print("TRAJECTORY REPLAY")
    print(f"  Poses: {len(arm_poses)}")
    print(f"  Speed factor: {speed_factor}x")
    print(f"  Loop: {loop}")
    print("="*60)
    print("\nPress Ctrl+C to stop playback\n")
    
    # Get initial transforms to convert body-frame poses to odom frame
    robot_state = robot_state_client.get_robot_state()
    
    # Calculate timing from timestamps
    timestamps = arm_poses[:, 0]
    start_time = timestamps[0]
    
    iteration = 0
    try:
        while True:
            iteration += 1
            print(f"\n--- Playback iteration {iteration} ---")
            
            playback_start = time.time()
            
            for i, pose in enumerate(arm_poses):
                # Extract pose data
                # Format: [timestamp, x, y, z, qx, qy, qz, qw, gripper_pct]
                ts = pose[0]
                x, y, z = pose[1], pose[2], pose[3]
                qx, qy, qz, qw = pose[4], pose[5], pose[6], pose[7]
                gripper_pct = pose[8] if len(pose) > 8 else 100.0
                
                # Calculate target time for this pose
                relative_time = (ts - start_time) / speed_factor
                target_time = playback_start + relative_time
                
                # Wait until it's time to execute this pose
                now = time.time()
                if target_time > now:
                    time.sleep(target_time - now)
                
                # Get current odom_T_body transform
                robot_state = robot_state_client.get_robot_state()
                odom_T_flat_body = get_a_tform_b(
                    robot_state.kinematic_state.transforms_snapshot,
                    ODOM_FRAME_NAME,
                    GRAV_ALIGNED_BODY_FRAME_NAME
                )
                
                # The recorded pose is in body frame, transform to odom frame
                body_T_hand = math_helpers.SE3Pose(
                    x, y, z,
                    math_helpers.Quat(qw, qx, qy, qz)
                )
                odom_T_hand = odom_T_flat_body * body_T_hand
                
                # Build arm pose command
                arm_command = RobotCommandBuilder.arm_pose_command(
                    odom_T_hand.x, odom_T_hand.y, odom_T_hand.z,
                    odom_T_hand.rot.w, odom_T_hand.rot.x,
                    odom_T_hand.rot.y, odom_T_hand.rot.z,
                    ODOM_FRAME_NAME,
                    seconds=0.5  # Time to reach pose
                )
                
                # Build gripper command
                gripper_fraction = gripper_pct / 100.0 if gripper_pct > 1.0 else gripper_pct
                gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(
                    gripper_fraction
                )
                
                # Combine and send command
                command = RobotCommandBuilder.build_synchro_command(arm_command, gripper_command)
                command_client.robot_command(command)
                
                # Progress update
                if i % 10 == 0:
                    elapsed = time.time() - playback_start
                    print(f"  Pose {i+1}/{len(arm_poses)} | "
                          f"Elapsed: {elapsed:.1f}s | "
                          f"Gripper: {gripper_fraction*100:.0f}%")
            
            print(f"\nPlayback complete! Duration: {time.time() - playback_start:.2f}s")
            
            if not loop:
                break
            
            print("Looping... (Ctrl+C to stop)")
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n\nPlayback interrupted by user")


def main():
    parser = argparse.ArgumentParser(description='Replay demonstration trajectory on Spot')
    bosdyn.client.util.add_base_arguments(parser)
    
    parser.add_argument('--demo-path', required=True,
                        help='Path to demonstration directory (containing arm_poses.npy)')
    parser.add_argument('--speed-factor', type=float, default=1.0,
                        help='Playback speed multiplier (default: 1.0)')
    parser.add_argument('--loop', action='store_true',
                        help='Loop the demonstration continuously')
    parser.add_argument('--skip-stand', action='store_true',
                        help='Skip the stand command (if robot already standing)')
    
    options = parser.parse_args()
    
    # Load demonstration
    print(f"\nLoading demonstration from: {options.demo_path}")
    try:
        arm_poses = load_demonstration(options.demo_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return False
    
    # Connect to robot
    print(f"\nConnecting to robot at {options.hostname}...")
    sdk = create_standard_sdk('TrajectoryReplayClient')
    robot = sdk.create_robot(options.hostname)
    
    try:
        bosdyn.client.util.authenticate(robot)
        robot.time_sync.wait_for_sync()
    except Exception as e:
        print(f"Failed to connect to robot: {e}")
        return False
    
    assert robot.has_arm(), 'Robot requires an arm to run this example.'
    
    # Setup clients
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    
    with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # Power on
        print("Powering on robot...")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), 'Robot power on failed.'
        print("Robot powered on.")
        
        # Stand
        if not options.skip_stand:
            print("Commanding robot to stand...")
            blocking_stand(command_client, timeout_sec=10)
            print("Robot standing.")
            
            # Unstow arm
            print("Unstowing arm...")
            unstow_command = RobotCommandBuilder.arm_ready_command()
            command_client.robot_command(unstow_command)
            time.sleep(2.0)
            print("Arm ready.")
        
        # Replay trajectory
        replay_trajectory(
            robot=robot,
            command_client=command_client,
            robot_state_client=robot_state_client,
            arm_poses=arm_poses,
            speed_factor=options.speed_factor,
            loop=options.loop
        )
        
        # Stow arm after replay
        print("\nStowing arm...")
        stow_command = RobotCommandBuilder.arm_stow_command()
        command_client.robot_command(stow_command)
        time.sleep(2.0)
        
        print("Done!")
        return True


if __name__ == '__main__':
    if not main():
        sys.exit(1)


# python real_spot_replay_demonstration.py --hostname 10.0.0.30 --demo-path manual_traj_records/demo_20260126_141153/ --speed-factor 0.5