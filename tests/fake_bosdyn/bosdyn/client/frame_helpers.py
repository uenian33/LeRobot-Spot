"""Frame names and lookups, enough for import and for simple transform queries."""

BODY_FRAME_NAME = "body"
DESIRED_TOOL_FRAME_NAME = "desired_tool"
GRAV_ALIGNED_BODY_FRAME_NAME = "flat_body"
HAND_FRAME_NAME = "hand"
ODOM_FRAME_NAME = "odom"
TOOL_FRAME_NAME = "tool"
VISION_FRAME_NAME = "vision"
WR1_FRAME_NAME = "arm0.link_wr1"


def get_a_tform_b(frame_tree_snapshot, frame_a, frame_b):
    """Return the a_tform_b transform, or None when the snapshot does not have it.

    Tests that care about a specific transform install their own snapshot object
    exposing `.get(frame_a, frame_b)`; everything else gets None, which is what
    the real helper returns for an unknown frame pair.
    """
    if frame_tree_snapshot is None:
        return None
    getter = getattr(frame_tree_snapshot, "get", None)
    if getter is None:
        return None
    return getter(frame_a, frame_b)
