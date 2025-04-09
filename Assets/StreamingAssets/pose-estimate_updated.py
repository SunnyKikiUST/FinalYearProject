import cv2
import sys
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
from multiprocessing import shared_memory

@torch.no_grad()
def run(poseweights="yolov7-w6-pose.pt",source="football1.mp4",device='cpu',view_img=True,
        save_conf=False,line_thickness = 3,hide_labels=False, hide_conf=True):
    print("hi 4")
    show_boundary = True
    isCalib = False
    # variables for push up or bridge count
    push_ups = 0
    direction = 0
    pose_start_time = None
    last_time = None
    total_time_success = None
    horizon_pos = "middle"
    vert_pos = "normal"
    errmsg = "None" #None, 2player or nokpt

    ## Choose Game Mode
    game_mode = 1  # 0 for normal, 1 for challenge, 2 for pause
    frame_count = 0  #count no of frames
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
    width, height = 640, 480
    channels = 3
   
    print("Starting reading shared memory")

    try:
      raise execption
    except Exception as e:
      print("leave system")
      #sys.exit(0)
      return

    try:
        # Connect to shared memory
        shm = shared_memory.SharedMemory(name=low_res_shm_name, create=False)
        
        # Create a NumPy array that maps to the shared memory
        frame_array = np.ndarray((height, width, channels), dtype=np.uint8, buffer=shm.buf)
        
        while True:
            # The array is directly mapped to shared memory, so it automatically updates
            # Convert from RGB to BGR for OpenCV display
            #frame = cv2.cvtColor(frame_array.copy(), cv2.COLOR_RGB2BGR)
            frame = frame_array.copy()
            
            frame_width = width  #get video frame width
            frame_height = height #get video frame height
            # recap the frame_width and frame_height(need to be multiple of 32)
            # frame_width = 1088 #can be changed
            # frame_height = 640 #can be changed

            # boundary to determine the horizonal and vertical position
            left_boundary = frame_width // 3
            right_boundary = 2 * (frame_width // 3) 
            upper_boundary = 60
            lower_boundary = 200 # distance calculated from top

            #calibration boxes
            calib_c1, calib_c2 = (left_boundary, 100), (right_boundary, 800)

            print(frame_width, frame_height)
            
            vid_write_image = letterbox(frame, (frame_width), stride=64, auto=True)[0] #init videowriter
            resize_height, resize_width = vid_write_image.shape[:2]
            out_video_name = f"{source.split('/')[-1].split('.')[0]}"
            out = cv2.VideoWriter(f"push_up_test.mp4",
                                cv2.VideoWriter_fourcc(*'mp4v'), 30,
                                (resize_width, resize_height))

            #while(cap.isOpened): #loop until cap opened or video not complete
            start_time = time.time()  # Start time for fps calculation
            #print("Frame {} Processing".format(frame_count+1))

            #ret, frame = cap.read()  #get frame and success from video capture
            frame = cv2.flip(frame, 1)
            
            #if ret: #if success is true, means frame exist
            orig_image = frame #store frame
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB) #convert frame to RGB
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
                        # if(n > 1):
                        #   game_mode = 2 # pause
                        #   isCalib = True
                        #   errmsg = "2player"
                    
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
                        
                        #Get kpt to determine position
                        head_x = kpts[0]
                        head_y = kpts[0+1]
                        head_z = kpts[0+2]
                        leg1_x = kpts[15*3]
                        leg1_z = kpts[15*3+2]
                        leg2_x = kpts[16*3]
                        leg2_z = kpts[16*3+2]

                        z_thres = 0.7 #consider the points only when their score is higher than this threshold

                        # if the score of the points are too low, pause and do calib again
                        if((leg1_z + leg2_z) < 0.5 or head_z < 0.25):
                          print("leg1_z confindence:", leg1_z)
                          print("leg2_z confindence:", leg2_z)
                          mode = 2
                          isCalib = True
                          errmsg = "nokpt"

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
                          horizon_pos = "left"
                        elif(right_count >= 3):
                          horizon_pos = "right"
                        elif(middle_count >= 3):
                          horizon_pos = "middle"
                        
                        vert_pos = ""
                        if(head_y < upper_boundary):
                          vert_pos = "jump"
                        elif(head_y > lower_boundary):
                          vert_pos = "crouch"
                        elif(upper_boundary < head_y < lower_boundary):
                          vert_pos = "normal"

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

                        #print position on image
                        cv2.putText(im0, horizon_pos, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
                        cv2.putText(im0, vert_pos, (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

                        #plot box for calib
                        #color in BGR
                        if isCalib:
                          if(c1[0]>calib_c1[0] and c1[1]>calib_c1[1]) and (c2[0]<calib_c2[0] and c2[1]<calib_c2[1]): # if player is within the given boxes
                            #color = (0,255,0) #???
                            isCalib = False
                          #else: # if outside the box
                          color = (0,238,0) # Green
                          plot_calib_box(im0, calib_c1, calib_c2, color=color,
                                      line_thickness=line_thickness + 3)

                        keypoints = kpts.view(-1, 3).cpu().numpy()
                        
                        ## Challenges
                        if game_mode == 1:
                          #if random.choice([True, False]):
                            #push up count
                            # push_ups, direction, _, _, _, _ = pushup_counter_from_keypoints(keypoints, push_ups, direction)
                            # cv2.putText(im0, str(push_ups), (right_boundary + 100, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
                          # else:
                            #bridge time count
                            _, pose_start_time, last_time, total_time_success = track_pose_duration(keypoints, pose_start_time, last_time, total_time_success)
                            cv2.putText(im0, str(total_time_success), (right_boundary + 100, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

                        if errmsg != "None":
                          print(f"error: {errmsg}")
                          errmsg = "None"

            # Calculate and display FPS
            fps = 1 / (time.time() - start_time)
            cv2.putText(im0, f"FPS: {fps:.2f}", (im0.shape[1] - 120, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # end_time = time.time()  #Calculatio for FPS
            # fps = 1 / (end_time - start_time)
            # total_fps += fps
            # frame_count += 1
            
            # fps_list.append(total_fps) #append FPS in list
            # time_list.append(end_time - start_time) #append time in list
            
            print(view_img)
            # Stream results
            if view_img:
              cv2.imshow("YOLOv7 Pose Estimation Demo", im0)
              cv2.waitKey(1)  # 1 millisecond
            out.write(im0)  #writing the video frame
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            
    except KeyboardInterrupt:
        print("Reader stopped by user.")
    except Exception as e:
        print(f"Error: {e}")

    # if source.isnumeric() :    
    #     cap = cv2.VideoCapture(int(source))    #pass video to videocapture object
    # else :
    #     cap = cv2.VideoCapture(source)    #pass video to videocapture object
   
    # if (cap.isOpened() == False):   #check if videocapture not opened
    #     print('Error while trying to read video. Please check path again')
    #     raise SystemExit()

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
    opt = parse_opt()
    strip_optimizer(opt.device,opt.poseweights)
    main(opt)

if __name__ == "__main__":
    process_main()
