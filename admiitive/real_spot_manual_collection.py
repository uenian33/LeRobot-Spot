#!/usr/bin/env python3
"""
Boston Dynamics Spot Arm Teleoperation Control
Supports keyboard control and ROS2-based 6DOF teleoperation
WITH SYNCHRONIZED RGB, DEPTH, AND POSE RECORDING
"""

import argparse
import logging
import os
import signal
import sys
import threading
import time
import math
import socket
import struct
import queue
from pathlib import Path

import numpy as np
import cv2

from google.protobuf import wrappers_pb2
from PIL import Image, ImageEnhance

import numpy as np
from scipy.spatial.transform import Rotation as R

# Boston Dynamics imports
import bosdyn.api.power_pb2 as PowerServiceProto
import bosdyn.api.robot_state_pb2 as robot_state_proto
import bosdyn.client.util
from bosdyn.api import arm_command_pb2, geometry_pb2, robot_command_pb2, synchronized_command_pb2
from bosdyn.client import ResponseError, RpcError, create_standard_sdk
from bosdyn.client.async_tasks import AsyncPeriodicQuery, AsyncPeriodicGRPCTask
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.lease import Error as LeaseBaseError
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.power import PowerClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.time_sync import TimeSyncError
from bosdyn.util import duration_str, secs_to_hms, seconds_to_duration
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b

# Image client
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient, build_image_request

# This script subclasses AsyncPeriodicQuery directly, so it needs the repair for
# the 5.1.x SDKs whose AsyncGRPCTask.update() shadows now_sec. The repo root is
# not on sys.path when this is run from admiitive/, and the package is not
# installed, so put it there.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lerobot_spot.bosdyn_compat import patch_async_task_update

patch_async_task_update()

# Configure logging
LOGGER = logging.getLogger()

# Control parameters
VELOCITY_CMD_DURATION = 0.15  # seconds
COMMAND_INPUT_RATE = 0.1  # seconds
VELOCITY_HAND_NORMALIZED = 0.2  # normalized hand velocity [0,1]
VELOCITY_ANGULAR_HAND = 0.2  # rad/sec
VELOCITY_BASE_SPEED = 0.4  # m/s
VELOCITY_BASE_ANGULAR = 0.6  # rad/sec

# Robot state query period
ROBOT_STATE_PERIOD = 0.2  # seconds

# Pose smoothing window
POSE_SMOOTHING_WINDOW = 5


def quaternion_to_euler(qx, qy, qz, qw):
    """Convert quaternion (qx, qy, qz, qw) to roll, pitch, yaw in radians."""
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.asin(sinp) if abs(sinp) <= 1 else math.copysign(math.pi / 2, sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def quaternion_difference(q1, q2):
    """
    Compute the difference quaternion between two quaternions.
    q1: target quaternion (qx, qy, qz, qw)
    q2: current quaternion (qx, qy, qz, qw)
    Returns: roll, pitch, yaw difference
    """
    q1x, q1y, q1z, q1w = q1
    q2x, q2y, q2z, q2w = q2

    # Inverse (conjugate) of q2
    q2_inv = (-q2x, -q2y, -q2z, q2w)

    # Compute difference quaternion: q_diff = q2_inv * q1
    q_diff_x = q2_inv[3] * q1x + q2_inv[0] * q1w + q2_inv[1] * q1z - q2_inv[2] * q1y
    q_diff_y = q2_inv[3] * q1y - q2_inv[0] * q1z + q2_inv[1] * q1w + q2_inv[2] * q1x
    q_diff_z = q2_inv[3] * q1z + q2_inv[0] * q1y - q2_inv[1] * q1x + q2_inv[2] * q1w
    q_diff_w = q2_inv[3] * q1w - q2_inv[0] * q1x - q2_inv[1] * q1y - q2_inv[2] * q1z

    # Convert to Euler angles
    return quaternion_to_euler(q_diff_x, q_diff_y, q_diff_z, q_diff_w)


class KeyboardListener:
    """Non-blocking keyboard listener using threading."""
    
    def __init__(self):
        self.key_queue = queue.Queue()
        self.running = False
        self.thread = None
        
        # Try to use termios for Unix systems
        try:
            import termios
            import tty
            self.use_termios = True
        except ImportError:
            self.use_termios = False
            print("Warning: termios not available, using basic input")
    
    def start(self):
        """Start the keyboard listener thread."""
        self.running = True
        self.thread = threading.Thread(target=self._listener_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the keyboard listener thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def _listener_loop(self):
        """Main listener loop."""
        if self.use_termios:
            self._termios_listener()
        else:
            self._basic_listener()
    
    def _termios_listener(self):
        """Unix termios-based non-blocking input."""
        import termios
        import tty
        import select
        
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while self.running:
                if select.select([sys.stdin], [], [], 0.01)[0]:
                    ch = sys.stdin.read(1)
                    if ch:
                        self.key_queue.put(ord(ch))
                else:
                    time.sleep(0.001)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    def _basic_listener(self):
        """Basic blocking input fallback."""
        while self.running:
            try:
                ch = sys.stdin.read(1)
                if ch:
                    self.key_queue.put(ord(ch))
            except:
                time.sleep(0.01)
    
    def get_key(self):
        """Get next key from queue (non-blocking)."""
        try:
            return self.key_queue.get_nowait()
        except queue.Empty:
            return -1


class ExitCheck:
    """Helper class to manage program exit signals."""

    def __init__(self):
        self._kill_now = False
        signal.signal(signal.SIGTERM, self._sigterm_handler)
        signal.signal(signal.SIGINT, self._sigterm_handler)

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        return False

    def _sigterm_handler(self, _signum, _frame):
        """Signal handler for SIGTERM/SIGINT."""
        self._kill_now = True

    def request_exit(self):
        """Manually trigger an exit."""
        self._kill_now = True

    @property
    def kill_now(self):
        """Return exit status."""
        return self._kill_now


class AsyncRobotState(AsyncPeriodicQuery):
    """Asynchronously grab robot state."""

    def __init__(self, robot_state_client):
        super(AsyncRobotState, self).__init__(
            'robot_state', 
            robot_state_client, 
            LOGGER,
            period_sec=ROBOT_STATE_PERIOD
        )

    def _start_query(self):
        return self._client.get_robot_state_async()


class TeleopServer:
    """TCP server to receive hand pose teleoperation data."""
    
    def __init__(self, port=5555):
        self.port = port
        self.running = False
        self.server_thread = None
        
        # Thread-safe data storage
        self.data_lock = threading.Lock()
        self.has_data = False
        self.position = np.zeros(3)
        self.rotation = np.zeros(3)
        self.gripper = 0.0
        self.message_count = 0
        
    def start(self):
        """Start the server thread."""
        self.running = True
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()
        
    def stop(self):
        """Stop the server thread."""
        self.running = False
        if self.server_thread:
            self.server_thread.join(timeout=2.0)
    
    def _server_loop(self):
        """Main server loop."""
        server_socket = None
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind(('0.0.0.0', self.port))
            server_socket.listen(1)
            server_socket.settimeout(1.0)
            
            while self.running:
                try:
                    client_socket, _ = server_socket.accept()
                    self._handle_client(client_socket)
                except socket.timeout:
                    continue
                except Exception:
                    pass
                    
        except Exception:
            pass
        finally:
            if server_socket:
                try:
                    server_socket.close()
                except:
                    pass
    
    def _handle_client(self, client_socket):
        """Handle client connection."""
        try:
            client_socket.settimeout(1.0)
            
            while self.running:
                try:
                    # Receive size header (4 bytes)
                    size_data = client_socket.recv(4)
                    if not size_data:
                        break
                    
                    array_size = struct.unpack('!I', size_data)[0]
                    
                    # Receive pose data
                    data = b''
                    while len(data) < array_size:
                        packet = client_socket.recv(array_size - len(data))
                        if not packet:
                            break
                        data += packet
                    
                    if len(data) != array_size:
                        break
                    
                    # Parse data (7 float64 values)
                    pose_data = np.frombuffer(data, dtype=np.float64)
                    
                    if len(pose_data) == 7:
                        with self.data_lock:
                            self.position = pose_data[0:3].copy()
                            self.rotation = pose_data[3:6].copy()
                            self.gripper = pose_data[6]
                            self.message_count += 1
                            self.has_data = True
                
                except socket.timeout:
                    continue
                except Exception:
                    break
        
        finally:
            try:
                client_socket.close()
            except:
                pass
    
    def get_data(self):
        """Get latest teleoperation data (thread-safe)."""
        with self.data_lock:
            return {
                'has_data': self.has_data,
                'position': self.position.copy() if self.has_data else None,
                'rotation': self.rotation.copy() if self.has_data else None,
                'gripper': self.gripper,
                'count': self.message_count
            }


class IntegratedArmControl:
    """Interface for controlling Spot's arm with keyboard and teleoperation."""

    def __init__(self, robot):
        # Robot clients
        self._robot = robot
        self._lease_client = robot.ensure_client(LeaseClient.default_service_name)
        
        # E-stop client (optional)
        try:
            self._estop_client = self._robot.ensure_client(EstopClient.default_service_name)
            self._estop_endpoint = EstopEndpoint(self._estop_client, 'GNClient', 9.0)
        except:
            self._estop_client = None
            self._estop_endpoint = None
            
        self._power_client = robot.ensure_client(PowerClient.default_service_name)
        self._robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
        self._robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        self._robot_image_client = robot.ensure_client(ImageClient.default_service_name)
        self._robot_state_task = AsyncRobotState(self._robot_state_client)
        
        # Threading
        self._lock = threading.Lock()
        self._exit_check = None
        
        # Command mapping
        self._command_dictionary = {
            27: self._stop,  # ESC key
            ord('\t'): self._quit_program,
            ord('T'): self._toggle_time_sync,
            ord(' '): self._toggle_estop,
            ord('P'): self._toggle_power,
            ord('v'): self._sit,
            ord('c'): self._stand,
            ord('w'): self._move_out,
            ord('s'): self._move_in,
            ord('a'): self._rotate_ccw,
            ord('d'): self._rotate_cw,
            ord('r'): self._move_up,
            ord('f'): self._move_down,
            ord('i'): self._rotate_plus_ry,
            ord('k'): self._rotate_minus_ry,
            ord('u'): self._rotate_plus_rx,
            ord('o'): self._rotate_minus_rx,
            ord('j'): self._rotate_plus_rz,
            ord('l'): self._rotate_minus_rz,
            ord('n'): self._toggle_gripper_open,
            ord('m'): self._toggle_gripper_closed,
            ord('y'): self._unstow,
            ord('h'): self._stow,
            ord('W'): self._body_forward,
            ord('S'): self._body_backward,
            ord('A'): self._body_left,
            ord('D'): self._body_right,
            ord('Q'): self._turn_left,
            ord('E'): self._turn_right,
            ord('x'): self._toggle_lease,
            ord('b'): self._save_recording,
            ord('z'): self._toggle_recording,
            ord('Z'): self._toggle_manual_mode,  # Manual demonstration collection
            ord('X'): self._replay_demonstration,  # Load and replay a saved trajectory
        }
        
        # Keyboard listener
        self.keyboard_listener = KeyboardListener()
        
        # UI state
        self._robot_id = None
        self._lease_keepalive = None
        self._estop_keepalive = None

        # Teleoperation server (receives hand pose data)
        self.teleop_server = TeleopServer(port=5555)

        # Teleoperation state
        self.ctrlr_init_pose = None
        self.arm_init_pose = None
        
        # Recording state (add after line ~289: self.status_print_interval = 2.0)
        self.recording = False
        self.recorded_poses = []
        self.recording_start_time = None

        # ====== NEW: Enhanced recording state for synchronized data ======
        self.recorded_rgb_images = []
        self.recorded_depth_images = []
        self.recorded_camera_poses = []
        self.recording_dir = None
        self.frame_counter = 0
        # Hand camera sources for Spot
        self.hand_color_source = "hand_color_image"
        self.hand_depth_source = "hand_depth_in_hand_color_frame"
        # ====== END NEW ======

        # Recording state
        self.rm_ctrl = False
        
        # Status display flag
        self.last_status_print = 0
        self.status_print_interval = 2.0  # Print status every 2 seconds

        # ====== Manual demonstration mode state ======
        self.manual_mode = False
        self.manual_recording_data = []
        self.manual_rgb_images = []
        self.manual_depth_images = []
        self.manual_camera_poses = []
        self.manual_frame_counter = 0
        self.manual_recording_dir = None
        self.manual_start_time = None
        self.manual_gripper_fraction = 1.0  # Gripper openness: 1.0 = fully open, 0.0 = closed
        
        # Replay state
        self.replaying = False
        self.replay_trajectory = None
        self.replay_index = 0

    def start(self):
        """Begin communication with robot."""
        print("Starting Spot Arm Control...")
        
        # Start keyboard listener
        self.keyboard_listener.start()
        
        # Start teleoperation server
        self.teleop_server.start()
        print("Teleoperation server started on port 5555")
        
        # Acquire lease
        try:    
            self._lease_keepalive = LeaseKeepAlive(
                self._lease_client, 
                must_acquire=True,
                return_at_exit=True
            )
            print("Lease acquired successfully")
        except:                                       
            self._lease_keepalive = self._lease_client.take()
            print("Lease forcibly taken successfully")

        self._robot_id = self._robot.get_id()
        print(f"Connected to robot: {self._robot_id.nickname} ({self._robot_id.serial_number})")
        
        # Setup E-stop if available
        if self._estop_endpoint is not None:
            self._estop_endpoint.force_simple_setup()
            print("E-stop endpoint configured")
        
        # Print initial status
        self._print_status()

    def shutdown(self):
        """Release control of robot gracefully."""
        print('\nShutting down ArmControl...')
        
        # Save recording if any data exists
        if len(self.recorded_poses) > 0:
            print("Saving recorded data before shutdown...")
            self._save_recording()
        
        # Stop keyboard listener
        self.keyboard_listener.stop()
        
        # Stop teleoperation server
        self.teleop_server.stop()
        
        # Shutdown E-stop and lease
        if self._estop_keepalive:
            self._estop_keepalive.shutdown()
        if self._lease_keepalive:
            try:
                self._lease_keepalive.shutdown()
            except Exception as e:
                print(f"Error shutting down lease: {e}")
        
        print("Shutdown complete")

    @property
    def robot_state(self):
        """Get latest robot state proto."""
        return self._robot_state_task.proto

    def _print_status(self):
        """Print current robot status."""
        print("\n" + "="*80)
        print(f"Robot: {self._robot_id.nickname} ({self._robot_id.serial_number})")
        print(f"Lease: {self._lease_str(self._lease_keepalive)}")
        print(f"Battery: {self._battery_str()}")
        print(f"E-stop: {self._estop_str()}")
        print(f"Power: {self._power_state_str()}")
        print(f"Time Sync: {self._time_sync_str()}")
        
        # Teleop status
        teleop_data = self.teleop_server.get_data()
        if teleop_data['has_data']:
            pos = teleop_data['position']
            rot = teleop_data['rotation']
            print(f"\nTeleop: ACTIVE (Messages: {teleop_data['count']})")
            print(f"  Position: X={pos[0]*100:7.2f} cm, Y={pos[1]*100:7.2f} cm, Z={pos[2]*100:7.2f} cm")
            print(f"  Rotation: Roll={np.degrees(rot[0]):7.2f}°, Pitch={np.degrees(rot[1]):7.2f}°, Yaw={np.degrees(rot[2]):7.2f}°")
            print(f"  Gripper: {teleop_data['gripper']:.2f}")
        else:
            print("\nTeleop: Waiting for data on port 5555")
        
        # ====== NEW: Recording status ======
        if self.recording:
            print(f"\nRecording: ACTIVE (Frames: {self.frame_counter})")
            if self.recording_dir:
                print(f"  Save directory: {self.recording_dir}")
        # ====== END NEW ======
        
        # Manual mode status
        if self.manual_mode:
            print(f"\nManual Mode: ACTIVE (Frames: {self.manual_frame_counter})")
            print(f"  Gripper: {'OPEN' if self.manual_gripper_fraction > 0.5 else 'CLOSED'} ({self.manual_gripper_fraction*100:.0f}%)")
        
        print("="*80 + "\n")

    def drive(self):
        """Main user interface control loop."""
        with ExitCheck() as self._exit_check:
            print("\n" + "="*80)
            print("SPOT ARM CONTROL - KEYBOARD COMMANDS")
            print("="*80)
            print("  [TAB]: Quit")
            print("  [T]: Time-sync, [SPACE]: E-stop, [P]: Power")
            print("  [c]: Stand, [v]: Sit")
            print("  [y]: Unstow arm, [h]: Stow arm")
            print("  [wasd]: Radial/Azimuthal control")
            print("  [rf]: Up/Down control")
            print("  [uo]: X-axis rotation control")
            print("  [ik]: Y-axis rotation control")
            print("  [jl]: Z-axis rotation control")
            print("  [nm]: Open/Close gripper")
            print("  [ESC]: Stop, [x]: Return/Acquire lease")
            print("  [nm]: Open/Close gripper")
            print("  [b]: Save recording, [z]: Toggle recording on/off")
            print("  [Z]: Toggle MANUAL demonstration mode (kinesthetic teaching)")
            print("       In manual mode: [n]=open gripper, [m]=close gripper")
            print("  [X]: REPLAY demonstration from hardcoded path (edit DEMO_REPLAY_PATH)")
            print("  [ESC]: Stop, [x]: Return/Acquire lease")
            print("  [WASDQE]: Body movement")
            print("="*80 + "\n")

            #try:
            while not self._exit_check.kill_now:
                self._robot_state_task.update()
                
                # Periodic status printing
                current_time = time.time()
                if current_time - self.last_status_print > self.status_print_interval:
                    self.last_status_print = current_time

                # Check teleoperation status
                teleop_data = self.teleop_server.get_data()
                if teleop_data['has_data']:
                    if not self.rm_ctrl:
                        print("\n>>> TELEOP MODE ACTIVATED <<<")
                        self.rm_ctrl = True
                else:
                    if self.rm_ctrl:
                        print("\n>>> TELEOP MODE DEACTIVATED <<<")
                        self.rm_ctrl = False

                #try:
                # Get keyboard input (non-blocking)
                cmd = self.keyboard_listener.get_key()
                if cmd != -1:
                    self._print_status()

                # Update teleoperation if active AND recording is enabled
                if self.rm_ctrl and self.recording:
                    self.update_robot_from_delta()

                # ====== Manual demonstration mode: send impedance cmd and record ======
                if self.manual_mode:
                    self._send_impedance_command()
                    self._record_manual_frame()

                # Process keyboard command
                if cmd != -1:
                    self._drive_cmd(cmd)
                
                time.sleep(COMMAND_INPUT_RATE)
                    
                #except Exception as e:
                #    print(f"\nERROR in control loop: {e}")
                #    import traceback
                #   traceback.print_exc()
                #    # On fault, safely power off
                #    self._safe_power_off()
                #    time.sleep(2.0)
                #    raise

            #except KeyboardInterrupt:
            #    print("\nKeyboard interrupt received")
            #except Exception as e:
            #    print(f"\nUnexpected error: {e}")
            #    import traceback
            #    traceback.print_exc()

    def _drive_cmd(self, key):
        """Execute user command based on key press."""
        try:
            cmd_function = self._command_dictionary[key]
            cmd_function()
        except KeyError:
            if key and key != -1 and key < 256:
                print(f'Unrecognized keyboard command: \'{chr(key)}\'')

    def _try_grpc(self, desc, thunk):
        """Try a gRPC call and handle errors."""
        try:
            return thunk()
        except (ResponseError, RpcError, LeaseBaseError) as err:
            print(f'Failed {desc}: {err}')
            return None

    def _try_grpc_async(self, desc, thunk):
        """Try an async gRPC call and handle errors."""
        def on_future_done(fut):
            try:
                fut.result()
            except (ResponseError, RpcError, LeaseBaseError) as err:
                print(f'Failed {desc}: {err}')
                return None

        future = thunk()
        future.add_done_callback(on_future_done)
    
    def _quit_program(self):
        """Quit the program gracefully."""
        print("\nQuitting program...")
        
        # Save recording if any data exists
        if len(self.recorded_poses) > 0:
            print("Saving recorded data before exit...")
            self._save_recording()
        
        self._sit()
        if self._exit_check is not None:
            self._exit_check.request_exit()

    # ========== Robot State Methods ==========
    
    def _toggle_recording(self):
        """Toggle recording on/off."""
        self.recording = not self.recording
        if self.recording:
            print("\n>>> RECORDING ENABLED (with RGB + Depth + Pose) <<<")
            self.recorded_poses = []
            self.recording_start_time = time.time()
            # ====== NEW: Reset enhanced recording state ======
            self.recorded_rgb_images = []
            self.recorded_depth_images = []
            self.recorded_camera_poses = []
            self.frame_counter = 0
            # Create recording directory with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.recording_dir = f"demo_recording_{timestamp}"
            os.makedirs(self.recording_dir, exist_ok=True)
            os.makedirs(os.path.join(self.recording_dir, "rgb"), exist_ok=True)
            os.makedirs(os.path.join(self.recording_dir, "depth"), exist_ok=True)
            print(f"  Recording directory: {self.recording_dir}")
            # ====== END NEW ======
        else:
            print("\n>>> RECORDING DISABLED <<<")
            print(f"Recorded {len(self.recorded_poses)} poses, {self.frame_counter} frames")

    def _save_recording(self):
        """Save recorded poses to numpy file."""
        if len(self.recorded_poses) == 0:
            print("No recorded data to save!")
            return
        
        # ====== MODIFIED: Use recording directory if available ======
        if self.recording_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.recording_dir = f"demo_recording_{timestamp}"
            os.makedirs(self.recording_dir, exist_ok=True)
        
        # Save arm poses
        poses_filename = os.path.join(self.recording_dir, "arm_poses.npy")
        recorded_array = np.array(self.recorded_poses)
        np.save(poses_filename, recorded_array)
        
        # ====== NEW: Save camera poses separately ======
        if len(self.recorded_camera_poses) > 0:
            camera_poses_filename = os.path.join(self.recording_dir, "camera_poses.npy")
            camera_poses_array = np.array(self.recorded_camera_poses)
            np.save(camera_poses_filename, camera_poses_array)
            print(f"  Saved camera poses: {camera_poses_array.shape}")
        
        # Save metadata
        metadata = {
            'num_frames': self.frame_counter,
            'num_poses': len(self.recorded_poses),
            'duration': time.time() - self.recording_start_time if self.recording_start_time else 0,
            'arm_pose_format': '[timestamp, x, y, z, qx, qy, qz, qw]',
            'camera_pose_format': '[timestamp, x, y, z, qx, qy, qz, qw] (hand frame in odom)',
            'rgb_format': 'PNG images in rgb/ directory',
            'depth_format': 'NPY arrays in depth/ directory (uint16, millimeters)',
        }
        metadata_filename = os.path.join(self.recording_dir, "metadata.npy")
        np.save(metadata_filename, metadata, allow_pickle=True)
        # ====== END NEW ======
        
        print(f"\n{'='*80}")
        print(f"Saved recording to {self.recording_dir}")
        print(f"  Arm poses: {recorded_array.shape}")
        print(f"  Frames: {self.frame_counter}")
        print(f"  Duration: {time.time() - self.recording_start_time:.2f} seconds" if self.recording_start_time else "")
        print(f"{'='*80}\n")
        
        # Reset recording
        self.recorded_poses = []
        self.recorded_rgb_images = []
        self.recorded_depth_images = []
        self.recorded_camera_poses = []
        self.frame_counter = 0
        self.recording = False
        self.recording_dir = None
        
    def _toggle_time_sync(self):
        """Toggle time synchronization."""
        if self._robot.time_sync.stopped:
            self._robot.start_time_sync()
            print("Time sync: STARTED")
        else:
            self._robot.time_sync.stop()
            print("Time sync: STOPPED")

    def _toggle_estop(self):
        """Toggle E-stop on/off."""
        if self._estop_client is not None and self._estop_endpoint is not None:
            if not self._estop_keepalive:
                self._estop_keepalive = EstopKeepAlive(self._estop_endpoint)
                print("E-stop: RELEASED")
            else:
                self._try_grpc('stopping estop', self._estop_keepalive.stop)
                self._estop_keepalive.shutdown()
                self._estop_keepalive = None
                print("E-stop: ENGAGED")
            # Print status after e-stop toggle
            self._print_status()

    def _toggle_lease(self):
        """Toggle lease acquisition."""
        if self._lease_client is not None:
            if self._lease_keepalive is None:
                self._lease_keepalive = LeaseKeepAlive(
                    self._lease_client, 
                    must_acquire=True,
                    return_at_exit=True
                )
                print("Lease: ACQUIRED")
            else:
                self._lease_keepalive.shutdown()
                self._lease_keepalive = None
                print("Lease: RELEASED")

    def _start_robot_command(self, desc, command_proto, end_time_secs=None):
        """Send a robot command."""
        def _start_command():
            self._robot_command_client.robot_command(
                command=command_proto,
                end_time_secs=end_time_secs
            )
        self._try_grpc(desc, _start_command)

    # ========== Basic Movement Commands ==========
    
    def _sit(self):
        """Sit the robot."""
        print("Command: SIT")
        self._start_robot_command('sit', RobotCommandBuilder.synchro_sit_command())

    def _stand(self):
        """Stand the robot and initialize arm pose."""
        print("Command: STAND")
        power_state = self._power_state()
        if power_state is None:
            print('Could not stand: power state is unknown')
            return

        if power_state == robot_state_proto.PowerState.STATE_ON:
            self._start_robot_command('stand', RobotCommandBuilder.synchro_stand_command())
            time.sleep(2.1)
            self._arm_set_init_pose()
        else:
            print("Cannot stand: robot is not powered on")

    def _stop(self):
        """Stop all robot motion."""
        print("Command: STOP")
        self._start_robot_command('stop', RobotCommandBuilder.stop_command())

    # Body motion commands
    def _body_forward(self):
        self._velocity_cmd_helper('move_forward', v_x=VELOCITY_BASE_SPEED)

    def _body_backward(self):
        self._velocity_cmd_helper('move_backward', v_x=-VELOCITY_BASE_SPEED)

    def _body_left(self):
        self._velocity_cmd_helper('strafe_left', v_y=VELOCITY_BASE_SPEED)

    def _body_right(self):
        self._velocity_cmd_helper('strafe_right', v_y=-VELOCITY_BASE_SPEED)

    def _turn_left(self):
        self._velocity_cmd_helper('turn_left', v_rot=VELOCITY_BASE_ANGULAR)

    def _turn_right(self):
        self._velocity_cmd_helper('turn_right', v_rot=-VELOCITY_BASE_ANGULAR)

    # Arm linear motion commands
    def _move_out(self):
        self._arm_cartesian_velocity_cmd_helper('move_out', v_x=VELOCITY_HAND_NORMALIZED)

    def _move_in(self):
        self._arm_cartesian_velocity_cmd_helper('move_in', v_x=-VELOCITY_HAND_NORMALIZED)

    def _rotate_ccw(self):
        self._arm_cartesian_velocity_cmd_helper('rotate_ccw', v_y=VELOCITY_HAND_NORMALIZED)

    def _rotate_cw(self):
        self._arm_cartesian_velocity_cmd_helper('rotate_cw', v_y=-VELOCITY_HAND_NORMALIZED)

    def _move_up(self):
        self._arm_cylindrical_velocity_cmd_helper('move_up', v_z=VELOCITY_HAND_NORMALIZED)

    def _move_down(self):
        self._arm_cylindrical_velocity_cmd_helper('move_down', v_z=-VELOCITY_HAND_NORMALIZED)

    # Arm rotation commands
    def _rotate_plus_rx(self):
        self._arm_angular_velocity_cmd_helper('rotate_plus_rx', v_rx=VELOCITY_ANGULAR_HAND)

    def _rotate_minus_rx(self):
        self._arm_angular_velocity_cmd_helper('rotate_minus_rx', v_rx=-VELOCITY_ANGULAR_HAND)

    def _rotate_plus_ry(self):
        self._arm_angular_velocity_cmd_helper('rotate_plus_ry', v_ry=VELOCITY_ANGULAR_HAND)

    def _rotate_minus_ry(self):
        self._arm_angular_velocity_cmd_helper('rotate_minus_ry', v_ry=-VELOCITY_ANGULAR_HAND)

    def _rotate_plus_rz(self):
        self._arm_angular_velocity_cmd_helper('rotate_plus_rz', v_rz=VELOCITY_ANGULAR_HAND)

    def _rotate_minus_rz(self):
        self._arm_angular_velocity_cmd_helper('rotate_minus_rz', v_rz=-VELOCITY_ANGULAR_HAND)

    # ========== Teleoperation Control ==========

        
    def apply_delta_rotation_(self, arm_init_pose, delta_pose):
        """
        Apply delta rotation from controller frame to robot frame.
        Args:
            arm_init_pose: [x, y, z, w] quaternion in robot frame
            delta_pose: [roll, pitch, yaw] in controller frame
        Returns:
            target_pose: [x, y, z, w] quaternion in robot frame
        """
        # Controller frame is rotated 90° clockwise (yaw right) relative to robot frame
        # This means controller's +Y points to robot's +X
        controller_to_robot = R.from_euler('z', 0, degrees=True)
        
        # Convert delta from controller frame to robot frame
        delta_rotation = R.from_euler('xyz', delta_pose, degrees=False)
        #delta_in_robot_frame = controller_to_robot * delta_rotation * controller_to_robot
        delta_in_robot_frame = controller_to_robot * delta_rotation * controller_to_robot
        
        # Apply delta to initial pose
        init_rotation = R.from_quat(arm_init_pose)
        # FIXED: Apply delta in robot frame, not compose with init rotation
        target_rotation = delta_in_robot_frame * init_rotation  # Changed order
        
        return target_rotation.as_quat()
    
      
    def apply_delta_rotation(self, arm_init_pose, delta_pose):
        """
        Apply delta rotation from controller frame to robot frame.
        Args:
            arm_init_pose: [x, y, z, w] quaternion in robot frame
            delta_pose: [roll, pitch, yaw] in controller frame
        Returns:
            target_pose: [x, y, z, w] quaternion in robot frame
        """
        
        #print("Delta Pose:", delta_pose)
        #print("Arm Init Pose:", arm_init_pose)

        arm_init_pose = R.from_quat(arm_init_pose).as_matrix()
        control_pose = R.from_euler('xyz', delta_pose, degrees=False).as_matrix()
        correction_matrix = R.from_euler('x', 0, degrees=True).as_matrix()
        corrected_control_pose = correction_matrix @ control_pose

        arm_to_body = corrected_control_pose @ arm_init_pose

        return R.from_matrix(arm_to_body).as_quat()

    # ====== NEW: Capture synchronized images from hand camera ======
    def _capture_hand_images(self):
        """
        Capture synchronized RGB and depth images from Spot's hand camera.
        Returns:
            tuple: (rgb_image, depth_image, capture_timestamp) or (None, None, None) on failure
        """
        try:
            # Build image requests for both RGB and depth
            image_requests = [
                build_image_request(
                    self.hand_color_source,
                    quality_percent=75,
                    image_format=image_pb2.Image.FORMAT_JPEG
                ),
                build_image_request(
                    self.hand_depth_source,
                    quality_percent=100,
                    image_format=image_pb2.Image.FORMAT_RAW
                )
            ]
            
            # Get images (synchronized request)
            image_responses = self._robot_image_client.get_image(image_requests)
            
            rgb_image = None
            depth_image = None
            capture_timestamp = time.time()
            
            for image_response in image_responses:
                if image_response.source.name == self.hand_color_source:
                    # Decode RGB image
                    rgb_data = np.frombuffer(image_response.shot.image.data, dtype=np.uint8)
                    rgb_image = cv2.imdecode(rgb_data, cv2.IMREAD_COLOR)
                    # Use image timestamp if available
                    if image_response.shot.acquisition_time.seconds > 0:
                        capture_timestamp = (
                            image_response.shot.acquisition_time.seconds + 
                            image_response.shot.acquisition_time.nanos * 1e-9
                        )
                        
                elif image_response.source.name == self.hand_depth_source:
                    # Decode depth image (raw format)
                    depth_data = np.frombuffer(image_response.shot.image.data, dtype=np.uint16)
                    height = image_response.shot.image.rows
                    width = image_response.shot.image.cols
                    depth_image = depth_data.reshape((height, width))
            
            return rgb_image, depth_image, capture_timestamp
            
        except Exception as e:
            print(f"Error capturing hand images: {e}")
            return None, None, None
    
    def _get_hand_camera_pose(self, robot_state):
        """
        Get the pose of the hand camera in the odom frame.
        Args:
            robot_state: Current robot state
        Returns:
            list: [x, y, z, qx, qy, qz, qw] or None on failure
        """
        try:
            # Get hand frame pose in odom frame
            odom_T_hand = get_a_tform_b(
                robot_state.kinematic_state.transforms_snapshot,
                ODOM_FRAME_NAME,
                "hand"  # hand frame is where the camera is mounted
            )
            
            return [
                odom_T_hand.position.x,
                odom_T_hand.position.y,
                odom_T_hand.position.z,
                odom_T_hand.rotation.x,
                odom_T_hand.rotation.y,
                odom_T_hand.rotation.z,
                odom_T_hand.rotation.w,
            ]
        except Exception as e:
            print(f"Error getting camera pose: {e}")
            return None
    
    def _get_camera_intrinsics(self):
        """
        Get camera intrinsics for the hand camera.
        Returns:
            dict: Camera intrinsics including fx, fy, cx, cy, width, height
        """
        try:
            # Get image sources to retrieve intrinsics
            sources = self._robot_image_client.list_image_sources()
            
            intrinsics = {}
            for source in sources:
                if source.name == self.hand_color_source:
                    if source.pinhole.HasField('intrinsics'):
                        intr = source.pinhole.intrinsics
                        intrinsics['rgb'] = {
                            'fx': intr.focal_length.x,
                            'fy': intr.focal_length.y,
                            'cx': intr.principal_point.x,
                            'cy': intr.principal_point.y,
                            'width': source.cols,
                            'height': source.rows,
                        }
                elif source.name == self.hand_depth_source:
                    if source.pinhole.HasField('intrinsics'):
                        intr = source.pinhole.intrinsics
                        intrinsics['depth'] = {
                            'fx': intr.focal_length.x,
                            'fy': intr.focal_length.y,
                            'cx': intr.principal_point.x,
                            'cy': intr.principal_point.y,
                            'width': source.cols,
                            'height': source.rows,
                        }
            
            return intrinsics if intrinsics else None
            
        except Exception as e:
            print(f"Error getting camera intrinsics: {e}")
            return None
    # ====== END NEW ======

    def update_robot_from_delta(self):
        """
        Update arm end effector position based on controller movement.
        This is the core teleoperation control loop.
        """
    
        # Get current arm state
        robot_state = self._robot_state_client.get_robot_state()
        gripper_angle = robot_state.manipulator_state.gripper_open_percentage
        current_arm_pose = robot_state.kinematic_state.transforms_snapshot\
            .child_to_parent_edge_map["hand"].parent_tform_child

        current_arm_state = [
            current_arm_pose.position.x,
            current_arm_pose.position.y,
            current_arm_pose.position.z,
            current_arm_pose.rotation.x,
            current_arm_pose.rotation.y,
            current_arm_pose.rotation.z,
            current_arm_pose.rotation.w,
        ]

        # ====== MODIFIED: Enhanced recording with synchronized images ======
        if self.rm_ctrl and self.recording:
            timestamp = time.time()
            
            # Capture synchronized RGB and depth images
            rgb_image, depth_image, img_timestamp = self._capture_hand_images()
            
            # Get camera pose in odom frame
            camera_pose = self._get_hand_camera_pose(robot_state)
            
            if rgb_image is not None and depth_image is not None:
                # Save RGB image
                rgb_filename = os.path.join(self.recording_dir, "rgb", f"frame_{self.frame_counter:06d}.png")
                cv2.imwrite(rgb_filename, rgb_image)
                
                # Save depth image as numpy array (preserves full 16-bit precision)
                depth_filename = os.path.join(self.recording_dir, "depth", f"frame_{self.frame_counter:06d}.npy")
                np.save(depth_filename, depth_image)
                
                # Store camera pose
                if camera_pose is not None:
                    camera_pose_data = [img_timestamp] + camera_pose
                    self.recorded_camera_poses.append(camera_pose_data)
                
                self.frame_counter += 1
                
                if self.frame_counter % 10 == 0:
                    print(f"Recording frame {self.frame_counter}...")
            
            # Store arm pose (original format)
            flat_pose = [
                timestamp,    
                current_arm_pose.position.x,
                current_arm_pose.position.y,
                current_arm_pose.position.z,
                current_arm_pose.rotation.x,
                current_arm_pose.rotation.y,
                current_arm_pose.rotation.z,
                current_arm_pose.rotation.w,
            ]
            self.recorded_poses.append(flat_pose)
        # ====== END MODIFIED ======

        # Get the pose offsets from the port 5555 
        teleop_data = self.teleop_server.get_data()
        
        if not teleop_data['has_data']:
            return

        # Get teleop data
        pos = teleop_data['position']
        rot = teleop_data['rotation']
        gripper_state = teleop_data['gripper']

        transformation = np.eye(4)
        transformation[0:3, 3] = pos
        rot_matrix = R.from_euler('xyz', rot, degrees=False).as_matrix()
        transformation[0:3, 0:3] = rot_matrix

        #print("input:", transformation[0:3, 3], R.from_matrix(transformation[0:3, 0:3]).as_euler('xyz', degrees=True))

        # Define the rotation matrix that performs the remapping
        # old z -> new -x:  new x-axis is [0, 0, -1] in old frame
        # old x -> new y:   new y-axis is [1, 0, 0] in old frame  
        # old y -> new -z:  new z-axis is [0, -1, 0] in old frame
        P = np.array([
            [0,  0, -1],
            [1,  0,  0],
            [0, -1,  0]
        ])

        # Extend to 4x4 homogeneous transformation
        P_homo = np.eye(4)
        P_homo[:3, :3] = P

        # Apply the rotation
        transformation = P_homo @ transformation @ P_homo.T

        pos = transformation[0:3, 3]
        original_rot = R.from_matrix(transformation[0:3, 0:3]).as_euler('xyz', degrees=False)

        rot[2] = -original_rot[0]
        rot[1] = -original_rot[1]
        rot[0] = -original_rot[2]

        arm_target_pose = R.from_euler('xyz', rot, degrees=False).as_rotvec() * 1.65
        arm_target_pose = R.from_rotvec(arm_target_pose).as_quat()

        #print("corrected:", transformation[0:3, 3], rot)

        # Position Coordinate transformation:
        # forward-backward +z -z  -----> spot: +x -x
        # right-left +x -x ------>  spot: -y +y
        # up-down +y -y -----> spot: +z -z 
        # 
        #rot[0]:pitch data   
        #rot[1]:roll data  
        #rot[2]: yaw data
        
        x_trans_motion_scale = 1.8
        y_trans_motion_scale = 1.8
        z_trans_motion_scale = 1.8
        delt_x, delt_y, delt_z = pos[0]*x_trans_motion_scale, \
                                 pos[1]*y_trans_motion_scale, \
                                 pos[2]*z_trans_motion_scale
        #delt_x, delt_y, delt_z = 0, 0, 0
        
        # FIXED: Correct rotation mapping for controller frame rotated 90° clockwise
        # Controller axes after 90° rotation: controller_Y → robot_X, controller_X → robot_-Y
        # When controller yaws right (+yaw), robot should yaw right (+yaw in robot frame)
        # When controller pitches up (+pitch), robot should pitch up (+pitch in robot frame)  
        # When controller rolls right (+roll), robot should roll right (+roll in robot frame)
        
        # rot[0]: pitch, rot[1]: roll, rot[2]: yaw (all in controller frame)
        controller_roll = rot[0]   # rotation around controller X axis
        controller_pitch = rot[1]  # rotation around controller Y axis
        controller_yaw = rot[2]    # rotation around controller Z axis
        
        # Because controller is rotated 90° clockwise relative to robot:
        # - Controller's pitch (around Y) becomes robot's roll (around X)
        # - Controller's roll (around X) becomes robot's pitch (around -Y)
        # - Controller's yaw (around Z) stays as robot's yaw (around Z)
        
        delta_rotation = [controller_roll, controller_pitch, controller_yaw]  # [roll, pitch, yaw] in controller frame
        
        delta_pose = np.array([delt_x, delt_y, delt_z, delta_rotation[0], delta_rotation[1], delta_rotation[2]])
        
        # Calculate target position by adding delta to initial arm pose
        target_position = [
            self.arm_init_pose[0] + delta_pose[0],  
            self.arm_init_pose[1] + delta_pose[1],  
            self.arm_init_pose[2] + delta_pose[2],  
        ]
        
        # Apply rotation with corrected transformation
        #arm_target_pose = self.apply_delta_rotation(self.arm_init_pose[3:], delta_pose[3:])
        #arm_target_pose = R.from_matrix(transformation[0:3, 0:3]).as_quat()
        
        
        #print(arm_target_pose, delta_pose[3:])
        # Record data
        current_arm_state.append(gripper_angle)

        # Build position command in gravity-aligned body frame
        hand_ewrt_flat_body = geometry_pb2.Vec3(
            x=target_position[0], 
            y=target_position[1], 
            z=target_position[2]
        )

        # Rotation as quaternion
        flat_body_Q_hand = geometry_pb2.Quaternion(
            w=arm_target_pose[3], 
            x=arm_target_pose[0], 
            y=arm_target_pose[1], 
            z=arm_target_pose[2]
        )

        flat_body_T_hand = geometry_pb2.SE3Pose(
            position=hand_ewrt_flat_body,
            rotation=flat_body_Q_hand
        )

        # Transform to odom frame
        robot_state = self._robot_state_client.get_robot_state()
        odom_T_flat_body = get_a_tform_b(
            robot_state.kinematic_state.transforms_snapshot,
            ODOM_FRAME_NAME, 
            GRAV_ALIGNED_BODY_FRAME_NAME
        )

        odom_T_hand = odom_T_flat_body * math_helpers.SE3Pose.from_proto(flat_body_T_hand)

        # Build and send arm command
        seconds = 0.3
        arm_command = RobotCommandBuilder.arm_pose_command(
            odom_T_hand.x, odom_T_hand.y, odom_T_hand.z, 
            odom_T_hand.rot.w, odom_T_hand.rot.x,
            odom_T_hand.rot.y, odom_T_hand.rot.z, 
            ODOM_FRAME_NAME, seconds
        )

        # Build gripper command based on gripper_state
        if gripper_state == 1:
            # Close gripper
            gripper_command = RobotCommandBuilder.claw_gripper_close_command()
        else:
            # Open gripper
            gripper_command = RobotCommandBuilder.claw_gripper_open_command()

        # Combine arm and gripper commands
        command = RobotCommandBuilder.build_synchro_command(arm_command, gripper_command)
        cmd_id = self._robot_command_client.robot_command(command)


        # Get feedback (optional, can comment out for less printing)
        # feedback_resp = self._robot_command_client.robot_command_feedback(cmd_id)
        # arm_feedback = feedback_resp.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback
        # measured_pos_dist = arm_feedback.measured_pos_distance_to_goal
        # measured_rot_dist = arm_feedback.measured_rot_distance_to_goal

    def _arm_set_init_pose(self):
        """Move arm to initial pose (in front of robot)."""
        print("Setting initial arm pose...")
        
        def block_until_arm_arrives_with_prints(robot, command_client, cmd_id):
            """Block until arm reaches goal with progress updates."""
            while True:
                feedback_resp = command_client.robot_command_feedback(cmd_id)
                arm_feedback = feedback_resp.feedback.synchronized_feedback\
                    .arm_command_feedback.arm_cartesian_feedback
                
                measured_pos_dist = arm_feedback.measured_pos_distance_to_goal
                measured_rot_dist = arm_feedback.measured_rot_distance_to_goal

                if arm_feedback.status == arm_command_pb2.ArmCartesianCommand.Feedback.STATUS_TRAJECTORY_COMPLETE:
                    print('Arm initialization complete.')
                    break
                time.sleep(0.1)

        # Build initial pose (in gravity-aligned body frame)
        x, y, z = 0.8, 0., 0.3
        hand_ewrt_flat_body = geometry_pb2.Vec3(x=x, y=y, z=z)

        # Rotation as quaternion (identity rotation)
        qw, qx, qy, qz = 1, 0, 0, 0
        flat_body_Q_hand = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)

        flat_body_T_hand = geometry_pb2.SE3Pose(
            position=hand_ewrt_flat_body,
            rotation=flat_body_Q_hand
        )

        # Transform to odom frame
        robot_state = self._robot_state_client.get_robot_state()
        odom_T_flat_body = get_a_tform_b(
            robot_state.kinematic_state.transforms_snapshot,
            ODOM_FRAME_NAME, 
            GRAV_ALIGNED_BODY_FRAME_NAME
        )

        odom_T_hand = odom_T_flat_body * math_helpers.SE3Pose.from_proto(flat_body_T_hand)

        # Build commands
        seconds = 2
        arm_command = RobotCommandBuilder.arm_pose_command(
            odom_T_hand.x, odom_T_hand.y, odom_T_hand.z, 
            odom_T_hand.rot.w, odom_T_hand.rot.x,
            odom_T_hand.rot.y, odom_T_hand.rot.z, 
            ODOM_FRAME_NAME, seconds
        )

        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)
        command = RobotCommandBuilder.build_synchro_command(gripper_command, arm_command)

        # Send command and wait
        cmd_id = self._robot_command_client.robot_command(command)
        block_until_arm_arrives_with_prints(self._robot, self._robot_command_client, cmd_id)

        arm_init_pose = self._robot_state_client.get_robot_state().kinematic_state.transforms_snapshot.child_to_parent_edge_map["hand"].parent_tform_child
        self.arm_init_pose = [arm_init_pose.position.x,
                              arm_init_pose.position.y,
                              arm_init_pose.position.z,
                              arm_init_pose.rotation.x,
                              arm_init_pose.rotation.y,
                              arm_init_pose.rotation.z,
                              arm_init_pose.rotation.w,] 
        
    # ========== Manual Demonstration Mode (Kinesthetic Teaching) ==========
    
    def _toggle_manual_mode(self):
        """Toggle manual demonstration mode on/off."""
        if not self.manual_mode:
            # Check if robot is standing and arm is initialized
            if self.arm_init_pose is None:
                print("ERROR: Please press 'c' to stand and initialize arm first!")
                return
            
            # Enter manual mode
            self.manual_mode = True
            self.manual_recording_data = []
            self.manual_rgb_images = []
            self.manual_depth_images = []
            self.manual_camera_poses = []
            self.manual_frame_counter = 0
            self.manual_start_time = time.time()
            self.manual_gripper_fraction = 1.0  # Start with gripper open
            
            # Create recording directory
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            os.makedirs("manual_traj_records", exist_ok=True)  # Ensure base dir exists
            self.manual_recording_dir = os.path.join("manual_traj_records", f"demo_{timestamp}")
            os.makedirs(self.manual_recording_dir, exist_ok=True)
            os.makedirs(os.path.join(self.manual_recording_dir, "rgb"), exist_ok=True)
            os.makedirs(os.path.join(self.manual_recording_dir, "depth"), exist_ok=True)
            
            print("\n" + "="*80)
            print(">>> MANUAL DEMONSTRATION MODE ACTIVATED <<<")
            print("  - Arm is now compliant (low stiffness)")
            print("  - Physically guide the arm to collect trajectory")
            print("  - Press [n] to OPEN gripper, [m] to CLOSE gripper")
            print("  - Press 'Z' again to stop and save")
            print(f"  - Recording to: {self.manual_recording_dir}")
            print("="*80 + "\n")
        else:
            # Exit manual mode and save
            self.manual_mode = False
            self._save_manual_recording()
            print("\n>>> MANUAL DEMONSTRATION MODE DEACTIVATED <<<")
    
    def _send_impedance_command(self):
        """Send low-stiffness impedance command to make arm compliant, with gripper control."""
        try:
            # Get current robot state
            robot_state = self._robot_state_client.get_robot_state()
            
            # Get odom_T_hand (current hand pose in odom frame)
            odom_T_hand = get_a_tform_b(
                robot_state.kinematic_state.transforms_snapshot,
                ODOM_FRAME_NAME,
                "hand"
            )
            
            # Use current hand position as task frame
            odom_T_task = math_helpers.SE3Pose(
                odom_T_hand.position.x,
                odom_T_hand.position.y,
                odom_T_hand.position.z,
                math_helpers.Quat(
                    odom_T_hand.rotation.w,
                    odom_T_hand.rotation.x,
                    odom_T_hand.rotation.y,
                    odom_T_hand.rotation.z
                )
            )
            
            # Identity tool frame (tool = hand frame)
            wr1_T_tool = math_helpers.SE3Pose(0, 0, 0, math_helpers.Quat(1, 0, 0, 0))
            
            # Build the impedance command (following official BD example pattern)
            robot_cmd = robot_command_pb2.RobotCommand()
            impedance_cmd = robot_cmd.synchronized_command.arm_command.arm_impedance_command
            
            # Set up frames
            impedance_cmd.root_frame_name = ODOM_FRAME_NAME
            impedance_cmd.root_tform_task.CopyFrom(odom_T_task.to_proto())
            impedance_cmd.wrist_tform_tool.CopyFrom(wr1_T_tool.to_proto())
            
            # Low stiffness for compliance (official max: 500 N/m linear, 60 Nm/rad rotational)
            # Setting very low values makes arm backdrivable
            impedance_cmd.diagonal_stiffness_matrix.CopyFrom(
                geometry_pb2.Vector(values=[40, 40, 40, 8, 8, 8]))
            
            # Damping for stability (official max: 2.5 linear, 1.0 rotational)
            impedance_cmd.diagonal_damping_matrix.CopyFrom(
                geometry_pb2.Vector(values=[1.0, 1.0, 1.0, 0.5, 0.5, 0.5]))
            
            # Set desired tool pose trajectory (identity = stay at task frame origin)
            # This is required by the API
            traj = impedance_cmd.task_tform_desired_tool
            pt1 = traj.points.add()
            pt1.time_since_reference.CopyFrom(seconds_to_duration(0.5))
            pt1.pose.CopyFrom(math_helpers.SE3Pose(0, 0, 0, math_helpers.Quat(1, 0, 0, 0)).to_proto())
            
            # ====== ADD GRIPPER COMMAND TO SAME SYNCHRONIZED COMMAND ======
            # Build gripper command using SDK builder
            gripper_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(
                self.manual_gripper_fraction
            )
            # Merge gripper into our synchronized command
            robot_cmd.synchronized_command.gripper_command.MergeFrom(
                gripper_cmd.synchronized_command.gripper_command
            )
            # ====== END GRIPPER COMMAND ======
            
            # Send combined command
            self._robot_command_client.robot_command(
                command=robot_cmd,
                end_time_secs=time.time() + 0.5
            )
            
        except Exception as e:
            print(f"Error sending impedance command: {e}")
            import traceback
            traceback.print_exc()
    
    def _record_manual_frame(self):
        """Record current arm state and images during manual mode."""
        try:
            timestamp = time.time()
            
            # Get current robot state
            robot_state = self._robot_state_client.get_robot_state()
            
            # Get arm pose
            current_arm_pose = robot_state.kinematic_state.transforms_snapshot\
                .child_to_parent_edge_map["hand"].parent_tform_child
            gripper_angle = robot_state.manipulator_state.gripper_open_percentage
            
            # Store arm pose
            pose_data = [
                timestamp,
                current_arm_pose.position.x,
                current_arm_pose.position.y,
                current_arm_pose.position.z,
                current_arm_pose.rotation.x,
                current_arm_pose.rotation.y,
                current_arm_pose.rotation.z,
                current_arm_pose.rotation.w,
                gripper_angle,
            ]
            self.manual_recording_data.append(pose_data)
            
            # Capture images (reuse existing method)
            rgb_image, depth_image, img_timestamp = self._capture_hand_images()
            
            if rgb_image is not None and depth_image is not None:
                # Save RGB image
                rgb_filename = os.path.join(
                    self.manual_recording_dir, "rgb", 
                    f"frame_{self.manual_frame_counter:06d}.png"
                )
                cv2.imwrite(rgb_filename, rgb_image)
                
                # Save depth image
                depth_filename = os.path.join(
                    self.manual_recording_dir, "depth",
                    f"frame_{self.manual_frame_counter:06d}.npy"
                )
                np.save(depth_filename, depth_image)
                
                # Get and store camera pose
                camera_pose = self._get_hand_camera_pose(robot_state)
                if camera_pose is not None:
                    self.manual_camera_poses.append([img_timestamp] + camera_pose)
                
                self.manual_frame_counter += 1
                
                if self.manual_frame_counter % 10 == 0:
                    print(f"  [Manual] Recording frame {self.manual_frame_counter}...")
                    
        except Exception as e:
            print(f"Error recording manual frame: {e}")
    
    def _save_manual_recording(self):
        """Save manual demonstration recording with full trajectory data."""
        if len(self.manual_recording_data) == 0:
            print("No manual demonstration data to save!")
            return
        
        # Save arm poses (trajectory)
        poses_filename = os.path.join(self.manual_recording_dir, "arm_poses.npy")
        poses_array = np.array(self.manual_recording_data)
        np.save(poses_filename, poses_array)
        
        # Save camera poses (trajectory)
        if len(self.manual_camera_poses) > 0:
            camera_poses_filename = os.path.join(self.manual_recording_dir, "camera_poses.npy")
            camera_poses_array = np.array(self.manual_camera_poses)
            np.save(camera_poses_filename, camera_poses_array)
        
        # Save camera intrinsics
        camera_intrinsics = self._get_camera_intrinsics()
        if camera_intrinsics is not None:
            intrinsics_filename = os.path.join(self.manual_recording_dir, "camera_intrinsics.npy")
            np.save(intrinsics_filename, camera_intrinsics, allow_pickle=True)
            print(f"  Saved camera intrinsics")
        
        # Save metadata
        duration = time.time() - self.manual_start_time if self.manual_start_time else 0
        metadata = {
            'num_frames': self.manual_frame_counter,
            'num_poses': len(self.manual_recording_data),
            'duration': duration,
            'recording_type': 'manual_kinesthetic',
            'arm_pose_format': '[timestamp, x, y, z, qx, qy, qz, qw, gripper_pct]',
            'camera_pose_format': '[timestamp, x, y, z, qx, qy, qz, qw] (hand frame in odom)',
            'rgb_format': 'PNG images in rgb/ directory',
            'depth_format': 'NPY arrays in depth/ directory (uint16, millimeters)',
            'camera_intrinsics_format': 'dict with rgb and depth keys, each containing fx, fy, cx, cy, width, height',
        }
        metadata_filename = os.path.join(self.manual_recording_dir, "metadata.npy")
        np.save(metadata_filename, metadata, allow_pickle=True)
        
        print("\n" + "="*80)
        print(f"MANUAL DEMONSTRATION SAVED")
        print(f"  Directory: {self.manual_recording_dir}")
        print(f"  Arm trajectory: {poses_array.shape}")
        if len(self.manual_camera_poses) > 0:
            print(f"  Camera trajectory: {np.array(self.manual_camera_poses).shape}")
        print(f"  Frames: {self.manual_frame_counter}")
        print(f"  Duration: {duration:.2f} seconds")
        print("="*80 + "\n")
        
        # Reset state
        self.manual_recording_data = []
        self.manual_rgb_images = []
        self.manual_depth_images = []
        self.manual_camera_poses = []
        self.manual_frame_counter = 0
        self.manual_recording_dir = None
        self.manual_start_time = None

    # ========== Trajectory Replay ==========
    
    # Hardcoded demonstration path for replay
    DEMO_REPLAY_PATH = "manual_traj_records/demo_20260126_144927"  # <-- Change this to your demo folder
    
    def _replay_demonstration(self):
        """Load and replay a saved demonstration trajectory."""
        if self.manual_mode:
            print("ERROR: Cannot replay while in manual mode. Press 'Z' to exit manual mode first.")
            return
        
        if self.arm_init_pose is None:
            print("ERROR: Please press 'c' to stand and initialize arm first!")
            return
        
        demo_path = self.DEMO_REPLAY_PATH
        
        print("\n" + "="*80)
        print("REPLAY DEMONSTRATION")
        print(f"Loading from: {demo_path}")
        print("="*80)
        
        # Check if path exists
        if not os.path.exists(demo_path):
            print(f"ERROR: Path does not exist: {demo_path}")
            print("Please update DEMO_REPLAY_PATH in the code.")
            return
        
        # Load arm trajectory
        arm_poses_file = os.path.join(demo_path, "arm_poses.npy")
        if not os.path.exists(arm_poses_file):
            print(f"ERROR: arm_poses.npy not found in {demo_path}")
            return
        
        try:
            trajectory = np.load(arm_poses_file)
            print(f"\nLoaded trajectory: {trajectory.shape}")
            print(f"  Format: [timestamp, x, y, z, qx, qy, qz, qw, gripper_pct]")
            print(f"  Duration: {trajectory[-1, 0] - trajectory[0, 0]:.2f} seconds")
            print(f"  Points: {len(trajectory)}")
        except Exception as e:
            print(f"ERROR: Failed to load trajectory: {e}")
            return
        
        print("\n>>> STARTING REPLAY <<<")
        
        # Execute trajectory
        self._execute_trajectory(trajectory)
        
        print("\n>>> REPLAY COMPLETE <<<")

    def _execute_trajectory(self, trajectory):
        """
        Execute a recorded arm trajectory.
        
        Args:
            trajectory: numpy array with shape (N, 9) where each row is
                       [timestamp, x, y, z, qx, qy, qz, qw, gripper_pct]
        """
        try:
            # Get timing from trajectory
            timestamps = trajectory[:, 0]
            start_time = time.time()
            trajectory_start = timestamps[0]
            
            current_idx = 0
            total_points = len(trajectory)
            
            print("  Press ESC or 'q' to abort replay, Ctrl+C to quit program")
            
            while current_idx < total_points:
                # Check for exit signal (Ctrl+C)
                if self._exit_check is not None and self._exit_check.kill_now:
                    print("\n>>> REPLAY INTERRUPTED (Ctrl+C) <<<")
                    self._stop()
                    return
                
                # Check for ESC or 'q' key to abort
                key = self.keyboard_listener.get_key()
                if key == 27 or key == ord('q'):  # ESC or 'q'
                    print("\n>>> REPLAY ABORTED <<<")
                    self._stop()
                    return
                
                # Calculate which trajectory point we should be at based on elapsed time
                elapsed = time.time() - start_time
                target_time = trajectory_start + elapsed
                
                # Find the trajectory point closest to current time
                while current_idx < total_points - 1 and timestamps[current_idx + 1] <= target_time:
                    current_idx += 1
                
                # Get pose data: [timestamp, x, y, z, qx, qy, qz, qw, gripper_pct]
                pose = trajectory[current_idx]
                x, y, z = pose[1], pose[2], pose[3]
                qx, qy, qz, qw = pose[4], pose[5], pose[6], pose[7]
                gripper_pct = pose[8] if len(pose) > 8 else 100.0
                
                # Get current robot state to transform pose
                robot_state = self._robot_state_client.get_robot_state()
                
                # The recorded pose is in body frame, transform to odom
                odom_T_body = get_a_tform_b(
                    robot_state.kinematic_state.transforms_snapshot,
                    ODOM_FRAME_NAME,
                    GRAV_ALIGNED_BODY_FRAME_NAME
                )
                
                # Build pose in body frame
                body_T_hand = math_helpers.SE3Pose(x, y, z, math_helpers.Quat(qw, qx, qy, qz))
                
                # Transform to odom frame
                odom_T_hand = odom_T_body * body_T_hand
                
                # Build arm command
                arm_command = RobotCommandBuilder.arm_pose_command(
                    odom_T_hand.x, odom_T_hand.y, odom_T_hand.z,
                    odom_T_hand.rot.w, odom_T_hand.rot.x,
                    odom_T_hand.rot.y, odom_T_hand.rot.z,
                    ODOM_FRAME_NAME,
                    seconds=0.2  # Short duration, will be updated frequently
                )
                
                # Build gripper command
                gripper_fraction = gripper_pct / 100.0
                gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(gripper_fraction)
                
                # Combine and send
                command = RobotCommandBuilder.build_synchro_command(arm_command, gripper_command)
                self._robot_command_client.robot_command(command)
                
                # Progress update
                if current_idx % 10 == 0:
                    progress = (current_idx + 1) / total_points * 100
                    print(f"  Replay progress: {progress:.1f}% ({current_idx + 1}/{total_points})", end="\r")
                
                # Small sleep to maintain roughly the same rate as recording
                time.sleep(COMMAND_INPUT_RATE)
            
            print(f"\n  Replay completed: {total_points} points")
            
        except Exception as e:
            print(f"\nError during replay: {e}")
            import traceback
            traceback.print_exc()
            self._stop()

    # ========== Velocity Command Helpers ==========
    
    def _velocity_cmd_helper(self, desc='', v_x=0.0, v_y=0.0, v_rot=0.0):
        """Send body velocity command."""
        self._start_robot_command(
            desc, 
            RobotCommandBuilder.synchro_velocity_command(v_x=v_x, v_y=v_y, v_rot=v_rot),
            end_time_secs=time.time() + VELOCITY_CMD_DURATION
        )

    def _arm_cartesian_velocity_cmd_helper(self, desc, v_x=0.0, v_y=0.0, v_z=0.0):
        """Send arm Cartesian velocity command."""
        cartesian_velocity = arm_command_pb2.ArmVelocityCommand.CartesianVelocity(
            frame_name='body'
        )
        cartesian_velocity.velocity_in_frame_name.x = v_x
        cartesian_velocity.velocity_in_frame_name.y = v_y
        cartesian_velocity.velocity_in_frame_name.z = v_z
            
        arm_velocity_command = arm_command_pb2.ArmVelocityCommand.Request(
            cartesian_velocity=cartesian_velocity,
            end_time=self._robot.time_sync.robot_timestamp_from_local_secs(
                time.time() + VELOCITY_CMD_DURATION
            )
        )

        self._arm_velocity_cmd_helper(arm_velocity_command=arm_velocity_command, desc=desc)

    def _arm_cylindrical_velocity_cmd_helper(self, desc='', v_r=0.0, v_theta=0.0, v_z=0.0):
        """Send arm cylindrical velocity command."""
        cylindrical_velocity = arm_command_pb2.ArmVelocityCommand.CylindricalVelocity()
        cylindrical_velocity.linear_velocity.r = v_r
        cylindrical_velocity.linear_velocity.theta = v_theta
        cylindrical_velocity.linear_velocity.z = v_z

        arm_velocity_command = arm_command_pb2.ArmVelocityCommand.Request(
            cylindrical_velocity=cylindrical_velocity,
            end_time=self._robot.time_sync.robot_timestamp_from_local_secs(
                time.time() + VELOCITY_CMD_DURATION
            )
        )

        self._arm_velocity_cmd_helper(arm_velocity_command=arm_velocity_command, desc=desc)

    def _arm_angular_velocity_cmd_helper(self, desc='', v_rx=0.0, v_ry=0.0, v_rz=0.0):
        """Send arm angular velocity command."""
        # Zero linear velocity
        cylindrical_velocity = arm_command_pb2.ArmVelocityCommand.CylindricalVelocity()

        # Build angular velocity command (expressed in hand frame, relative to odom)
        angular_velocity_of_hand_rt_odom_in_hand = geometry_pb2.Vec3(
            x=v_rx, y=v_ry, z=v_rz
        )

        arm_velocity_command = arm_command_pb2.ArmVelocityCommand.Request(
            cylindrical_velocity=cylindrical_velocity,
            angular_velocity_of_hand_rt_odom_in_hand=angular_velocity_of_hand_rt_odom_in_hand,
            end_time=self._robot.time_sync.robot_timestamp_from_local_secs(
                time.time() + VELOCITY_CMD_DURATION
            )
        )

        self._arm_velocity_cmd_helper(arm_velocity_command=arm_velocity_command, desc=desc)

    def _arm_velocity_cmd_helper(self, arm_velocity_command, desc=''):
        """Helper to send arm velocity commands."""
        robot_command = robot_command_pb2.RobotCommand()
        robot_command.synchronized_command.arm_command.arm_velocity_command.CopyFrom(
            arm_velocity_command
        )

        self._start_robot_command(
            desc, 
            robot_command,
            end_time_secs=time.time() + VELOCITY_CMD_DURATION
        )

    # ========== Gripper Commands ==========
    
    def _toggle_gripper_open(self):
        """Open the gripper."""
        # If in manual mode, update the gripper fraction instead of sending separate command
        if self.manual_mode:
            self.manual_gripper_fraction = 1.0  # Fully open
            print(f"  [Manual] Gripper: OPEN (fraction={self.manual_gripper_fraction})")
            return
        
        print("Command: OPEN GRIPPER")
        self._start_robot_command(
            'open_gripper', 
            RobotCommandBuilder.claw_gripper_open_command()
        )

    def _toggle_gripper_closed(self):
        """Close the gripper."""
        # If in manual mode, update the gripper fraction instead of sending separate command
        if self.manual_mode:
            self.manual_gripper_fraction = 0.0  # Fully closed
            print(f"  [Manual] Gripper: CLOSED (fraction={self.manual_gripper_fraction})")
            return
        
        print("Command: CLOSE GRIPPER")
        self._start_robot_command(
            'close_gripper', 
            RobotCommandBuilder.claw_gripper_close_command()
        )

    def _stow(self):
        """Stow the arm."""
        print("Command: STOW ARM")
        self._start_robot_command('stow', RobotCommandBuilder.arm_stow_command())

    def _unstow(self):
        """Unstow (ready) the arm."""
        print("Command: UNSTOW ARM")
        self._start_robot_command('unstow', RobotCommandBuilder.arm_ready_command())

    # ========== Power Control ==========
    
    def _toggle_power(self):
        """Toggle robot power on/off."""
        power_state = self._power_state()
        if power_state is None:
            print('Could not toggle power: power state is unknown')
            return

        if power_state == robot_state_proto.PowerState.STATE_OFF:
            print("Powering ON robot...")
            self._try_grpc_async('powering-on', self._request_power_on)
        else:
            print("Powering OFF robot...")
            self._try_grpc('powering-off', self._safe_power_off)
        
        # Print status after power toggle
        time.sleep(1.0)
        self._print_status()

    def _request_power_on(self):
        """Request power on."""
        request = PowerServiceProto.PowerCommandRequest.REQUEST_ON
        return self._power_client.power_command_async(request)

    def _safe_power_off(self):
        """Safely power off the robot."""
        self._start_robot_command(
            'safe_power_off', 
            RobotCommandBuilder.safe_power_off_command()
        )

    def _power_state(self):
        """Get current power state."""
        state = self.robot_state
        if not state:
            return None
        return state.power_state.motor_power_state

    # ========== Status String Methods ==========
    
    def _lease_str(self, lease_keep_alive):
        """Format lease status string."""
        if lease_keep_alive is None:
            alive = 'STOPPED'
            lease = 'RETURNED'
        else:
            try:
                _lease = lease_keep_alive.lease_wallet.get_lease()
                lease = f'{_lease.lease_proto.resource}:{_lease.lease_proto.sequence}'
            except LeaseBaseError:
                lease = '...'
            alive = 'RUNNING' if lease_keep_alive.is_alive() else 'STOPPED'
        return f'{lease} THREAD:{alive}'

    def _power_state_str(self):
        """Format power state string."""
        power_state = self._power_state()
        if power_state is None:
            return 'Unknown'
        state_str = robot_state_proto.PowerState.MotorPowerState.Name(power_state)
        return state_str[6:]  # Remove STATE_ prefix

    def _estop_str(self):
        """Format E-stop status string."""
        if not self._estop_client:
            thread_status = 'NOT ESTOP'
        else:
            thread_status = 'RUNNING' if self._estop_keepalive else 'STOPPED'
            
        estop_status = '??'
        state = self.robot_state
        if state:
            for estop_state in state.estop_states:
                if estop_state.type == estop_state.TYPE_SOFTWARE:
                    estop_status = estop_state.State.Name(estop_state.state)[6:]  # Remove STATE_
                    break
        return f'{estop_status} (thread: {thread_status})'

    def _time_sync_str(self):
        """Format time sync status string."""
        if not self._robot.time_sync:
            return '(none)'
            
        if self._robot.time_sync.stopped:
            status = 'STOPPED'
            exception = self._robot.time_sync.thread_exception
            if exception:
                status = f'{status} Exception: {exception}'
        else:
            status = 'RUNNING'
            
        try:
            skew = self._robot.time_sync.get_robot_clock_skew()
            skew_str = f'offset={duration_str(skew)}' if skew else '(Skew undetermined)'
        except (TimeSyncError, RpcError) as err:
            skew_str = f'({err})'
            
        return f'{status} {skew_str}'

    def _battery_str(self):
        """Format battery status string."""
        if not self.robot_state:
            return 'Unknown'
            
        battery_state = self.robot_state.battery_states[0]
        status = battery_state.Status.Name(battery_state.status)
        status = status[7:]  # Remove STATUS_ prefix
        
        if battery_state.charge_percentage.value:
            bar_len = int(battery_state.charge_percentage.value) // 10
            bat_bar = f'|{"=" * bar_len}{" " * (10 - bar_len)}|'
        else:
            bat_bar = ''
            
        time_left = ''
        if battery_state.estimated_runtime:
            time_left = f'({secs_to_hms(battery_state.estimated_runtime.seconds)})'
            
        return f'{status}{bat_bar} {time_left}'


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument(
        '--time-sync-interval-sec',
        help='Time-sync update interval (seconds)',
        type=float
    )
    options = parser.parse_args()

    # Setup logging to print to console
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    # Create robot object
    sdk = create_standard_sdk('WASDClient')
    robot = sdk.create_robot(options.hostname)
    
    try:
        robot.authenticate('rllab', 'robotlearninglab')
        bosdyn.client.util.authenticate(robot)
        robot.start_time_sync(options.time_sync_interval_sec)
    except RpcError as err:
        print(f'Failed to communicate with robot: {err}')
        return False
    
    assert robot.has_arm(), 'Robot requires an arm to run this example.'

    arm_control = IntegratedArmControl(robot)
    
    try:
        try:
            arm_control.start()
        except (ResponseError, RpcError) as err:
            print(f'Failed to initialize robot communication: {err}')
            return False

        try:
            arm_control.drive()
        except Exception as e:
            print(f'Arm control has thrown an error: {e}')
            import traceback
            traceback.print_exc()
        finally:
            arm_control.shutdown()

    except KeyboardInterrupt:
        print('\nCtrl+C detected')
        arm_control.shutdown()
    finally:
        print('Finished')
        if 'arm_control' in locals():
            arm_control.shutdown()
        
    return True


if __name__ == '__main__':
    if not main():
        sys.exit(1)