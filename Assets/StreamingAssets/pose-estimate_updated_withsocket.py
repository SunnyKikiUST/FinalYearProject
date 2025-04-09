import cv2
import time
import torch
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt,strip_optimizer,xyxy2xywh
from utils.plots import output_to_keypoint, plot_skeleton_kpts,colors,plot_one_box_kpt,plot_calib_box
from utils.push_up import pushup_counter_from_keypoints, bridge_counter_from_keypoints, track_pose_duration
from socket import *
import struct
import threading
from multiprocessing import shared_memory
from model_common_utils import listen_for_exit_signal

HOST = '127.0.0.1'
PORT = 65451

@torch.no_grad()
def run(poseweights="yolov7-w6-pose.pt",source="football1.mp4",device='cpu',view_img=True,
        save_conf=False,line_thickness = 3,hide_labels=False, hide_conf=True):
    show_boundary = True
    isCalib = False
    # variables for push up and bridge count
    push_ups = 0
    direction = 0
    pose_start_time = None
    horizon_pos = "middle" 
    horizon_pos_num = 1 # 0: left, 1: middle, 2: right
    vert_pos = "normal"
    vert_pos_num = 0 # 0: normal, 1: crouch, 2: jump 
    errmsg = "None" #None, 2player or nokpt

    ## Choose Game Mode
    game_mode = 0  # 0 for normal, 1 for challenge, 2 for pause due to loss of keypoints, 3 for pause due to multiple players detection, 4 for pause when game start in order to capture all keypoints 
    frame_count = 0  #count no of frames
    is_push_up = -1 # store the exercise that is going to do in exercise challenge
    total_fps = 0  #count total fps
    time_list = []   #list to store time
    fps_list = []    #list to store fps
    
    device = select_device(device) #select device
    half = device.type != 'cpu'

    model = attempt_load(poseweights, map_location=device)  #Load model
    _ = model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names

    # Preparation of using shared memory for getting low resolution frame
    low_res_shm_name = "fitness_fat_camera_low_res"
    frame_width, frame_height = 640, 480
    channels = 3
    
    try:
        print("Starting reading shared memory")
        # Connect to shared memory
        while True:
          try:
              shm = shared_memory.SharedMemory(name=low_res_shm_name, create=False)
              break
          except FileNotFoundError:
              print("Shared memory not found. Waiting...")
              time.sleep(1)

        #Socket communcation setup for communicating with C# game section
        server_socket = socket(AF_INET, SOCK_STREAM) # Sunny
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print("Waiting for Unity to connect pose estimation model...")

        client_socket, addr = server_socket.accept() # Accept connect when cap is opened
        print(f"Pose estimation model: Connected to Unity at {addr}") # Sunny

        threading.Thread(target=listen_for_exit_signal, args=(client_socket,), daemon = True) # Sunny

        # boundary to determine the horizonal and vertical position
        left_boundary = frame_width // 3
        right_boundary = 2 * (frame_width // 3) 
        upper_boundary = 60
        lower_boundary = 200 # distance calculated from top

        #calibration boxes
        calib_c1, calib_c2 = (left_boundary, 100), (right_boundary, 800)

        print(frame_width, frame_height)

    
        ######## Tell player to get ready by asking them to move into the green container when the game starter
        ######## game mode 3 will only show up when the game starts
        game_mode = 4
        isCalib = True

        # out = cv2.VideoWriter(f"ingame_test.mp4",
        #             cv2.VideoWriter_fourcc(*'mp4v'), 30,
        #             (frame_width, frame_height))

        while(True): 
            image = np.ndarray((frame_height, frame_width, channels), dtype=np.uint8, buffer=shm.buf) # get image from shared memory
            image = cv2.flip(image, 1)
            image = letterbox(image, (frame_width), stride=64, auto=True)[0]
            image_ = image.copy()
            image = transforms.ToTensor()(image)
            image = torch.tensor(np.array([image.numpy()]))
        
            image = image.to(device)  #convert image data to device
            image = image.float() #convert image to float precision (cpu)
            start_time = time.time() #start time for fps calculation
        
            with torch.no_grad():  #get predictions
                output_data, _ = model(image)
            output_data = non_max_suppression_kpt(output_data,   #Apply non max suppression
                                        0.25,   # Conf. Threshold.
                                        0.65, # IoU Threshold.
                                        nc=model.yaml['nc'], # Number of classes.
                                        nkpt=model.yaml['nkpt'], # Number of keypoints.
                                        kpt_label=True)
        
            output = output_to_keypoint(output_data)

            im0 = image[0].permute(1, 2, 0) * 255 # Change format [b, c, h, w] to [h, w, c] for displaying the image.
            im0 = im0.cpu().numpy().astype(np.uint8)
            
            im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR) #reshape image format to (BGR)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            for i, pose in enumerate(output_data):  # detections per image

                if len(output_data):  #check if no pose
                    for c in pose[:, 5].unique(): # Print results
                        n = (pose[:, 5] == c).sum()  # detections per class
                        
                        print("No of Objects in Current Frame : {}".format(n))
                        
                        # pause and do calib when more than 1 player detected
                        if(n > 1) and game_mode != 4 and game_mode != 1:
                          game_mode = 3 if game_mode != 4 else 4
                          isCalib = True
                          errmsg = "2player"
                    
                    for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:,:6])): #loop over poses for drawing on frame
                        c = int(cls)  # integer class
                        kpts = pose[det_index, 6:]
                        #Get middle corner points [5,6,11,12]
                        #bottom legs 15, 16
                        #nose 0

                        corner_index = [5, 6, 11, 12]
                        pts = []
                        for i in corner_index:
                          x = (kpts[i*3])
                          y = (kpts[i*3+1])
                          z = (kpts[i*3+2])
                          pts.append((x, y, z))
                        x = [p[0] for p in pts]
                        y = [p[1] for p in pts]
                        z = [p[2] for p in pts]
                        #Calculate center pt
                        center = (sum(x) / len(pts), sum(y) / len(pts))
                        score = sum(z) / len(pts)
                        # Add new kpt with score the average of pts
                        if(device == torch.device('cuda:0')):
                          center = torch.tensor([center[0], center[1], score], device=torch.device('cuda:0'))
                          kpts = torch.cat((kpts, center), 0)
                        else:
                          center = torch.tensor([center[0], center[1], score])
                          kpts = torch.cat((kpts, center), 0)

                        #plot boundary boxes
                        if show_boundary:
                          # draw horizontal boundary
                          cv2.line(im0, (left_boundary, 0), (left_boundary, frame_height), (0,0,255), 1 )
                          cv2.line(im0, (right_boundary, 0), (right_boundary, frame_height), (0,0,255), 1 )
                          # draw vertical boundary
                          cv2.line(im0, (0, upper_boundary), (frame_width, upper_boundary), (0,0,255), 1 )
                          cv2.line(im0, (0, lower_boundary), (frame_width, lower_boundary), (0,0,255), 1 )


                        #plot boxes of the player
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        c1, c2 = plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True), 
                                    line_thickness=line_thickness, kpt_label=True, kpts=kpts, steps=3, 
                                    orig_shape=im0.shape[:2])
                        print("c1:", c1)
                        print("c2:", c2)

                        ############################ Different Game Mode ######################################## 
                        ## Basic Movement Mode (the code section is for game_mode 0, 2, 3
                        if game_mode != 1:  
                          #Get kpt to determine position
                          head_x = kpts[0]
                          head_y = kpts[0+1]
                          head_z = kpts[0+2]
                          leg1_x = kpts[15*3] # left ankle
                          leg1_z = kpts[15*3+2]
                          leg2_x = kpts[16*3] # right ankle
                          leg2_z = kpts[16*3+2]

                          z_thres = 0.7 #consider the points only when their score is higher than this threshold

                          # if the score of the points are too low, pause and do calib again
                          if((leg1_z + leg2_z) < 0.5 or head_z < 0.25):
                            game_mode = 2 if game_mode != 4 else 4
                            isCalib = True
                            errmsg = "nokpt"

                          # For determine position
                          boundary_kpts = [(head_x,head_z), (center[0], score), (leg1_x,leg1_z), (leg2_x,leg2_z)]
                          left_count = 0
                          right_count = 0
                          middle_count = 0
                          for kpt in boundary_kpts:
                            if(kpt[0] < left_boundary and kpt[1] > z_thres):
                              left_count += 1
                            elif(kpt[0] > right_boundary and kpt[1] > z_thres):
                              right_count += 1
                            elif(left_boundary < kpt[0] < right_boundary and kpt[1] > z_thres):
                              middle_count += 1
                          
                          
                          # Decide the position when points in that region >= 3
                          if(left_count >= 3):
                            horizon_pos, horizon_pos_num = "left", 0 #left
                          elif(right_count >= 3):
                            horizon_pos, horizon_pos_num = "right", 2 #right
                          elif(middle_count >= 3):
                            horizon_pos, horizon_pos_num = "middle", 1 #middle
                          
                          vert_pos = ""
                          if(head_y < upper_boundary):
                            vert_pos, vert_pos_num = "jump", 2
                          elif(head_y > lower_boundary):
                            vert_pos, vert_pos_num = "crouch", 1
                          elif(upper_boundary < head_y < lower_boundary):
                            vert_pos, vert_pos_num = "normal", 0

                          #print position on image
                          cv2.putText(im0, horizon_pos, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
                          cv2.putText(im0, vert_pos, (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)



                          #plot box for calib
                          #color in BGR
                          if isCalib:
                            if(c1[0]>calib_c1[0] and c1[1]>calib_c1[1]) and (c2[0]<calib_c2[0] and c2[1]<calib_c2[1] and n == 1): # if player is within the given boxes
                              isCalib = False
                              print("player in inside the container!!!!!!!!!!")
                              game_mode = 0
                            #else: # if outside the box or have multiple players
                            color = (0,238,0) # Green
                            plot_calib_box(im0, calib_c1, calib_c2, color=color,
                                        line_thickness=line_thickness + 3)
                        
                        ## Challenges Mode
                        elif game_mode == 1:
                          print("inside game mode 1", push_ups)
                          # if is_push up == -1:
                          #   is_push_up =  random.choice([0, 1])
                          is_push_up = 1

                          keypoints = kpts.view(-1, 3).cpu().numpy()
                          if is_push_up == 1:
                            #push up count
                            push_ups, direction, _, _, _, _ = pushup_counter_from_keypoints(keypoints, push_ups, direction)
                            print(f"push up times:{push_ups}")
                            cv2.putText(im0, str(push_ups), (right_boundary + 100, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
                          else:
                            #bridge time count
                            _, pose_start_time, last_time, total_time_success = track_pose_duration(keypoints, pose_start_time, last_time, total_time_success)
                            cv2.putText(im0, str(total_time_success), (right_boundary + 100, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

                        ## Error Mode
                        if game_mode == 2 or game_mode == 3:
                          print(f"error: {errmsg}")
                        ## Gaming Starting Posture Capture Mode

                        ##################################################################
                    
                    # Must send game_mode in each loop
                    enc_game_mode = struct.pack("i", game_mode)
                    client_socket.sendall(enc_game_mode) 

                    # Calculate and display FPS
                    fps = 1 / (time.time() - start_time)
                    cv2.putText(im0, f"FPS: {fps:.2f}", (im0.shape[1] - 120, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    if game_mode == 0:
                      require_exercise_challenge = client_socket.recv(1) # To receive the signal of requesting exercise challenge
                      if int.from_bytes(require_exercise_challenge, byteorder='big') == 1: # Change to exercise challenge mode
                        game_mode = 1

                      # Send horizontal detection result (left, middle or right) 
                      enc_position_num = struct.pack("i", horizon_pos_num) 
                      client_socket.sendall(enc_position_num) 

                      # Send vertical detection result (normal, crouch or jump ) 
                      enc_position_num = struct.pack("i", vert_pos_num) 
                      client_socket.sendall(enc_position_num) 

                      # Encode frame as jpg
                      encoded, buffer = cv2.imencode('.jpg', im0) 

                      message_size = struct.pack("L", len(buffer)) # Converts the integer into a bytes-like object (L : 4 BYTES)
                      client_socket.sendall(message_size) 
                      # Send frame
                      client_socket.sendall(buffer) 
                    
                    # Challenge mode
                    elif game_mode == 1:
                      if is_push_up == 1: # is push up
                        print("in game mode 1 loop - push up")
                        enc_push_ups = struct.pack("f", push_ups) 
                        client_socket.sendall(enc_push_ups) 
                      else: # is bridge
                        print("in game mode 1 loop - bridge")
                        enc_total_time_success = struct.pack("f", total_time_success) 
                        client_socket.sendall(enc_total_time_success) 

                      # Encode frame as jpg
                      encoded, buffer = cv2.imencode('.jpg', im0) 

                      message_size = struct.pack("L", len(buffer)) # Converts the integer into a bytes-like object (L : 4 BYTES)
                      client_socket.sendall(message_size) 
                      # Send frame
                      client_socket.sendall(buffer) 

                      # should be byte to boolean
                      require_exercise_challenge = client_socket.recv(1) # To receive the signal of requesting exercise challenge


                      if int.from_bytes(require_exercise_challenge, byteorder='big') == 0: # Change to normal mode
                        game_mode = 0
                        push_ups = 0
                        total_time_success = 0


                    elif game_mode == 2 or game_mode == 3 or game_mode == 4:
                      # Encode frame as jpg
                      encoded, buffer = cv2.imencode('.jpg', im0) 

                      message_size = struct.pack("L", len(buffer)) # Converts the integer into a bytes-like object (L : 4 BYTES)
                      client_socket.sendall(message_size) 
                      # Send frame
                      client_socket.sendall(buffer) 

            # Stream results
            # if view_img:
            #   cv2.imshow("YOLOv7 Pose Estimation Demo", im0)
            #   cv2.waitKey(1)  # 1 millisecond
            #out.write(im0)  #writing the video frame

        # # cv2.destroyAllWindows()
        # avg_fps = total_fps / frame_count
        # print(f"Average FPS: {avg_fps:.3f}")
        
        # #plot the comparision graph
        # plot_fps_time_comparision(time_list=time_list,fps_list=fps_list)
    except KeyboardInterrupt:
        print("Socket stopped by user.")
    except Exception as e:
        print(f"Error: {e}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='0', help='video/0 for webcam') #video source

    # Check for CUDA availability and set default device 
    default_device = '0' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', type=str, default=default_device, help='cpu/0,1,2,3(gpu)')   #device arugments

    parser.add_argument('--view-img', default=True, action='store_true', help='display results')  #display results
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels') #save confidence in txt writing
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)') #box linethickness
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels') #box hidelabel
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences') #boxhideconf
    opt = parser.parse_args()
    return opt

#function for plot fps and time comparision graph
def plot_fps_time_comparision(time_list,fps_list):
    plt.figure()
    plt.xlabel('Time (s)')
    plt.ylabel('FPS')
    plt.title('FPS and Time Comparision Graph')
    plt.plot(time_list, fps_list,'b',label="FPS & Time")
    plt.savefig("FPS_and_Time_Comparision_pose_estimate.png")
    

#main function
def main(opt):
    run(**vars(opt))

def process_main():
    print("pose model test 1")
    opt = parse_opt()
    print("pose model test 2")
    strip_optimizer(opt.device,opt.poseweights)
    print("pose model test 3")
    main(opt)

if __name__ == "__main__":
    process_main()