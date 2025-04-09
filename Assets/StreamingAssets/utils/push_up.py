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

#This one is for haed towarding the webcam. can be easily be cheated.
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
        print("in push up function 1")
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
        left_percent = np.interp(angle_left, (90, 135), (100, 0))
        right_percent = np.interp(angle_right, (100, 135), (100, 0))
        print("in push up function 2")
        # Push-up logic
        if left_percent == 100 and right_percent == 100:
            if direction == 0:
                push_ups += 0.5
                direction = 1
        elif left_percent == 0 and right_percent == 0:
            if direction == 1:
                push_ups += 0.5
                direction = 0
        print("in push up function 3")
        return push_ups, direction, angle_left, angle_right, left_percent, right_percent

    except IndexError:
        # Handle cases where keypoints are incomplete or missing
        return push_ups, direction, None, None, None, None



# def bridge_counter_from_keypoints(keypoints, hip_angle_range=(150, 185), knee_angle_range=(60, 90)):

#   try:
#     right_shoulder = np.array(keypoints[6][:2])
#     right_hip = np.array(keypoints[12][:2])
#     right_knee = np.array(keypoints[14][:2])
#     right_ankle = np.array(keypoints[16][:2])

#     hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
#     knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

#     return (
#         hip_angle_range[0] <= hip_angle <= hip_angle_range[1] and
#         knee_angle_range[0] <= knee_angle <= knee_angle_range[1]
#     )


#   except IndexError:

#     return None

def bridge_counter_from_keypoints(keypoints, hip_angle_range=(150, 185), knee_angle_range=(60, 90), 
                                 min_confidence=0.3):
    try:
        # Extract both sides with confidence values
        # Right side
        right_points = {
            "shoulder": (np.array(keypoints[6][:2]), keypoints[6][2]),
            "hip": (np.array(keypoints[12][:2]), keypoints[12][2]),
            "knee": (np.array(keypoints[14][:2]), keypoints[14][2]),
            "ankle": (np.array(keypoints[16][:2]), keypoints[16][2])
        }
        
        # Left side
        left_points = {
            "shoulder": (np.array(keypoints[5][:2]), keypoints[5][2]),
            "hip": (np.array(keypoints[11][:2]), keypoints[11][2]),
            "knee": (np.array(keypoints[13][:2]), keypoints[13][2]),
            "ankle": (np.array(keypoints[15][:2]), keypoints[15][2])
        }
        
        # Check validity of both sides by considering confidence
        right_valid = all(conf >= min_confidence for _, conf in right_points.values())
        left_valid = all(conf >= min_confidence for _, conf in left_points.values())
        
        results = []
        
        # Calculate right side if valid
        if right_valid:
            r_shoulder, r_hip, r_knee, r_ankle = [p[0] for p in right_points.values()]
            right_hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)
            right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
            
            right_in_range = (
                hip_angle_range[0] <= right_hip_angle <= hip_angle_range[1] and
                knee_angle_range[0] <= right_knee_angle <= knee_angle_range[1]
            )
            
            # Calculate average confidence for right side
            right_conf = sum(conf for _, conf in right_points.values()) / 4
            results.append((right_in_range, right_conf))
        
        # Calculate left side if valid
        if left_valid:
            l_shoulder, l_hip, l_knee, l_ankle = [p[0] for p in left_points.values()]
            left_hip_angle = calculate_angle(l_shoulder, l_hip, l_knee)
            left_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
            
            left_in_range = (
                hip_angle_range[0] <= left_hip_angle <= hip_angle_range[1] and
                knee_angle_range[0] <= left_knee_angle <= knee_angle_range[1]
            )
            
            # Calculate average confidence for left side
            left_conf = sum(conf for _, conf in left_points.values()) / 4
            results.append((left_in_range, left_conf))
        
        # No valid sides
        if not results:
            return None
            
        # If only one side is valid, use it
        if len(results) == 1:
            return results[0][0]
            
        # If both sides are valid but disagree, use the side with higher confidence
        if results[0][0] != results[1][0]:
            return results[0][0] if results[0][1] > results[1][1] else results[1][0]
            
        # Both sides agree
        return results[0][0]
        
    except (IndexError, TypeError) as e:
        print(f"Error in bridge detection: {e}")
        return None

# hold_time=-1 mean hold_time is not used and only the accumulated time is needed
# If holding bridge continuous is important, then use elapsed, if total time of holding but continuous is not important, then use total_time_success
def track_pose_duration(keypoints, start_time=None, last_time=None, total_time_success=None, hold_time=-1):
    is_pose = bridge_counter_from_keypoints(keypoints)
    current_time = time.time()
    
    if is_pose:
        # If this is the first time detecting the pose or coming back from the pose lost
        if start_time is None:
            start_time = current_time
            if total_time_success == None: # Do not set to 0 if total_time_success is accumlated already
                total_time_success = 0 
        else:
            # Calculate time since last check
            elapsed = current_time - last_time
            total_time_success += elapsed  
            
        # Update last time for next calculation of accumlated time in next loop
        last_time = current_time
        
        # Check if pose has been held long enough ( This one only will use if hold_time is used)
        if hold_time != -1 and total_time_success >= hold_time:
            return 1, start_time, last_time, total_time_success  # Success
        else:
            return (-1), start_time, last_time, total_time_success  # Ongoing
    else:
        # Pose lost - reset start_time but keep accumulated time (i.e. total_time_success) if needed
        return (-1), None, None, total_time_success  # Reset timer and accumulated time
