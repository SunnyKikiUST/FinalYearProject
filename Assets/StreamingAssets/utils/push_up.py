import numpy as np

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
    Count push-ups using keypoint data.
    Args:
        keypoints: A numpy array containing keypoint data in COCO format.
        push_ups: Current push-up count (default: 0).
        direction: Current push-up direction, 0 for down and 1 for up (default: 0).
    Returns:
        Updated push-up count and direction.
    """
    try:
        # Extract required keypoints (COCO format indices)
        # left_shoulder = np.array(keypoints[11][:2])
        # right_shoulder = np.array(keypoints[12][:2])
        # left_elbow = np.array(keypoints[13][:2])
        # right_elbow = np.array(keypoints[14][:2])
        # left_wrist = np.array(keypoints[15][:2])
        # right_wrist = np.array(keypoints[16][:2])
        
        left_shoulder = np.array(keypoints[5][:2])
        right_shoulder = np.array(keypoints[6][:2])
        left_elbow = np.array(keypoints[7][:2])
        right_elbow = np.array(keypoints[8][:2])
        left_wrist = np.array(keypoints[9][:2])
        right_wrist = np.array(keypoints[10][:2])

        # Calculate angles for left and right arms
        angle_left = calculate_angle(left_shoulder, left_elbow, left_wrist)
        angle_right = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # Map angles to percentages
        left_percent = np.interp(angle_left, (90, 150), (100, 0))
        right_percent = np.interp(angle_right, (90, 150), (100, 0))

        # Push-up logic
        if left_percent == 100 and right_percent == 100:
            if direction == 0:
                push_ups += 0.5
                direction = 1
        elif left_percent == 0 and right_percent == 0:
            if direction == 1:
                push_ups += 0.5
                direction = 0

        return push_ups, direction, angle_left, angle_right, left_percent, right_percent

    except IndexError:
        # Handle cases where keypoints are incomplete or missing
        return push_ups, direction, None, None, None, None