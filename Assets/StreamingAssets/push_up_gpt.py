import numpy as np
import time

def calculate_angle(p1, p2, p3):
    """
    Calculate the angle between three points using the cosine rule.
    """
    a = np.linalg.norm(p2 - p3)
    b = np.linalg.norm(p1 - p3)
    c = np.linalg.norm(p1 - p2)
    angle = np.degrees(np.arccos((a**2 + c**2 - b**2) / (2 * a * c)))
    return angle

def pushup_counter_from_keypoints(keypoints, push_ups=0, direction=0):
    """
    Count push-ups using keypoint data, considering one arm and hip angle for form.
    Args:
        keypoints: A numpy array containing keypoint data in COCO format.
        push_ups: Current push-up count (default: 0).
        direction: Current push-up direction, 0 for down and 1 for up (default: 0).
    Returns:
        Updated push-up count, direction, arm angle, hip angle, arm percentage, hip status.
    """
    try:

        # Extract left and right shoulder confidence scores
        left_shoulder_conf = keypoints[5][2]
        right_shoulder_conf = keypoints[6][2]

        # Determine which side to track based on shoulder visibility
        if left_shoulder_conf > right_shoulder_conf:
            selected_side = 'left'
            shoulder_kp = keypoints[5]
            elbow_kp = keypoints[7]
            wrist_kp = keypoints[9]
            hip_kp = keypoints[11]
            knee_kp = keypoints[13]
            angle_range = (90, 160)  # Adjusted for better range
        else:
            selected_side = 'right'
            shoulder_kp = keypoints[6]
            elbow_kp = keypoints[8]
            wrist_kp = keypoints[10]
            hip_kp = keypoints[12]
            knee_kp = keypoints[14]
            angle_range = (140, 160)  # Unified range for both arms

        # Check confidence of required keypoints (threshold can be adjusted)
        min_conf = 0.2
        required_confs = [
            shoulder_kp[2], elbow_kp[2], wrist_kp[2],
            hip_kp[2], knee_kp[2]
        ]
        if any(conf < min_conf for conf in required_confs):
            return push_ups, direction, None, None, None, None

        # Extract coordinates
        shoulder = np.array(shoulder_kp[:2])
        elbow = np.array(elbow_kp[:2])
        wrist = np.array(wrist_kp[:2])
        hip = np.array(hip_kp[:2])
        knee = np.array(knee_kp[:2])

        # Calculate arm angle and percentage
        arm_angle = calculate_angle(shoulder, elbow, wrist)
        arm_percent = np.interp(arm_angle, angle_range, (100, 0))
        arm_percent = np.clip(arm_percent, 0, 100)

        # Calculate hip angle (shoulder, hip, knee)
        hip_angle = calculate_angle(shoulder, hip, knee)
        hip_threshold = 150  # Adjust threshold based on requirements
        hip_status = hip_angle >= hip_threshold if not np.isnan(hip_angle) else False

        # Push-up logic with hip angle check
        if direction == 0:
            if arm_percent >= 95 and hip_status:
                push_ups += 0.5
                direction = 1
        else:
            if arm_percent <= 5 and hip_status:
                push_ups += 0.5
                direction = 0

        return push_ups, direction, arm_angle, hip_angle, selected_side, hip_status

    except (IndexError, TypeError, ValueError):
        # Handle cases where keypoints are missing or invalid
        return push_ups, direction, None, None, None, None

    except IndexError:
        # Handle cases where keypoints are incomplete or missing
        return push_ups, direction, None, None, None, None



def bridge_counter_from_keypoints(keypoints, hip_angle_range=(150, 185), knee_angle_range=(60, 90)):

  try:
    right_shoulder = np.array(keypoints[6][:2])
    right_hip = np.array(keypoints[12][:2])
    right_knee = np.array(keypoints[14][:2])
    right_ankle = np.array(keypoints[16][:2])

    hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
    knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

    return (
        hip_angle_range[0] <= hip_angle <= hip_angle_range[1] and
        knee_angle_range[0] <= knee_angle <= knee_angle_range[1]
    )


  except IndexError:

    return None

def track_pose_duration(keypoints, start_time=None, hold_time=20):
    is_pose = bridge_counter_from_keypoints(keypoints)
    
    if is_pose:
        if start_time is None:
            start_time = time.time()  # Initialize timer
        elapsed = time.time() - start_time
        if elapsed >= hold_time:
            return 1, start_time, elapsed  # Pose held for 5 seconds
        else:
            return (-1), start_time, elapsed  # Pose ongoing but not yet 5s
    else:
        return (-1), None, None  # Reset timer
