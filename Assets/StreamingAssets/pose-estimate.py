import cv2
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt,strip_optimizer,xyxy2xywh
from utils.plots import output_to_keypoint, plot_skeleton_kpts,colors,plot_one_box_kpt,plot_calib_box
#from utils.push_up import pushup_counter_from_keypoints # Sunny
from socket import *
import struct
import threading
from queue import Queue

HOST = '127.0.0.1'
PORT = 65451

def listen_for_exit_signal(client_socket): # Sunny
  """
  Thread function to listen for the exit signal from Unity.
  """
  global should_exit
  while True:
    try:
      data = client_socket.recv(1024)
      if data:
        message = data.decode('utf-8').strip()
        if message == "exit":
            print("Termination signal received. Exiting.")
            should_exit = True
            break
    except Exception as e:
        print(f"Error in socket thread: {e}")
        break 

@torch.no_grad()
def run(poseweights="yolov7-w6-pose.pt",source="0",device='cpu',view_img=False,
        save_conf=False,line_thickness = 3,hide_labels=False, hide_conf=True):
    # variables for push up count
    #push_ups = 0
    #direction = 0

    frame_count = 0  #count no of frames
    # total_fps = 0  #count total fps
    # time_list = []   #list to store time
    # fps_list = []    #list to store fps
    should_exit = False
    
    device = select_device(opt.device) #select device
    half = device.type != 'cpu'
    if half:
      model = attempt_load(poseweights, map_location=device).half()  #Load model
    else:
      model = attempt_load(poseweights, map_location=device)  #Load model
    _ = model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
   
    if source.isnumeric() :    
        cap = cv2.VideoCapture(int(source), cv2.CAP_MSMF)    #pass video to videocapture object
    else :
        cap = cv2.VideoCapture(source)    #pass video to videocapture object
   
    if (cap.isOpened() == False):   #check if videocapture not opened
        print('Error while trying to read video. Please check path again')
        raise SystemExit()
    else:
        #Socket communcation setup for communicating with C# game section
        server_socket = socket(AF_INET, SOCK_STREAM) # Sunny
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print("Waiting for Unity to connect...")
        ret, frame = cap.read() # Let camera operate before real task execution
        client_socket, addr = server_socket.accept() # Accept connect when cap is opened
        print(f"Connected to Unity at {addr}") # Sunny

        threading.Thread(target=listen_for_exit_signal, args=(client_socket,), daemon = True) # Sunny

        frame_width = int(cap.get(3)) #get video frame width
        if frame_width > 1280: # Sunny
          frame_width = 1280

        while(cap.isOpened): #loop until cap opened or video not complete
            #print("Frame {} Processing".format(frame_count+1)) # Sunny

            start_time = time.time() #start time for fps calculation, Sunny change position of the code
            ret, frame = cap.read()  #get frame and success from video capture
            if should_exit:
              print("Game termination")
              break

            if ret: #if success is true, means frame exist (remove by Sunny)
              orig_image = frame #store frame
              image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB) #convert frame to RGB
              image = letterbox(image, (frame_width), stride=64, auto=True)[0]
              image_ = image.copy()
              image = transforms.ToTensor()(image)
              image = torch.tensor(np.array([image.numpy()]))
          
              image = image.to(device)  #convert image data to device
              image = image.float() #convert image to float precision (cpu)
          
              with torch.no_grad():  #get predictions
                  output_data, _ = model(image)
              output_data = non_max_suppression_kpt(output_data,   #Apply non max suppression
                                          0.25,   # Conf. Threshold.
                                          0.65, # IoU Threshold.
                                          nc=model.yaml['nc'], # Number of classes.
                                          nkpt=model.yaml['nkpt'], # Number of keypoints.
                                          kpt_label=True)
          
              #output = output_to_keypoint(output_data)
              

              im0 = image[0].permute(1, 2, 0) * 255 # Change format [b, c, h, w] to [h, w, c] for displaying the image.
              im0 = im0.cpu().numpy().astype(np.uint8)
              
              im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR) #reshape image format to (BGR)
              gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
              position_num = -1  # Sunny

              for i, pose in enumerate(output_data):  # detections per image
              
                  if len(output_data):  #check if no pose
                      for c in pose[:, 5].unique(): # Print results
                          n = (pose[:, 5] == c).sum()  # detections per class
                          
                          print("No of Objects in Current Frame : {}".format(n))
                      
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
                          
                          #Get Head kpt
                          head_x = kpts[0]
                          head_z = kpts[0+2]
                          leg1_x = kpts[15*3]
                          leg1_z = kpts[15*3+2]
                          leg2_x = kpts[16*3]
                          leg2_z = kpts[16*3+2]

                          z_thres = 0.7 #consider the points only when their score is higher than this threshold

                          # boundary to determine the horizonal position
                          left_boundary = frame_width // 3 # Edit by Sunny
                          right_boundary = 2 * (frame_width // 3)  # Edit by Sunny



                          boundary_kpts = [(head_x,head_z), (center[0], score), (leg1_x,leg1_z), (leg2_x,leg2_z)]
                          left_count = 0
                          right_count = 0
                          for kpt in boundary_kpts:
                            if(kpt[0] < left_boundary and kpt[1] > z_thres):
                              left_count += 1
                            elif(kpt[0] > right_boundary and kpt[1] > z_thres):
                              right_count += 1
                          
                          
                          # Decide the position when points in that region >= 3
                          position = None
                          if(left_count >= 3): # Sunny Edited
                            position, position_num = "right", 2
                          elif(right_count >= 3):
                            position, position_num = "left", 0
                          else:
                            position, position_num = "middle", 1

                          
                          label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                          c1, c2 = plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True), 
                                      line_thickness=opt.line_thickness,kpt_label=True, kpts=kpts, steps=3, 
                                      orig_shape=im0.shape[:2])

                          #plot box for calib
                          #color in BGR
                          cv2.putText(im0, position, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA) # Sunny
                          cv2.line(im0, (left_boundary, 0), (left_boundary, im0.shape[0]), (0, 255, 0), thickness=2)  # Left boundary
                          cv2.line(im0, (right_boundary, 0), (right_boundary, im0.shape[0]), (0, 0, 255), thickness=2)  # Sunny

                      # Send detection result (left, middle or right) 
                      enc_position_num = struct.pack("i", position_num) # Sunny
                      client_socket.sendall(enc_position_num) # Sunny

                      # Encode frame as jpg
                      encoded, buffer = cv2.imencode('.jpg', im0) # Sunny

                      message_size = struct.pack("L", len(buffer)) # Converts the integer into a bytes-like object (L : 4 BYTES)
                      client_socket.sendall(message_size) # Sunny
                      # Send frame
                      client_socket.sendall(buffer) # Sunny

                  # cv2.imshow("YOLOv7 Pose Estimation Demo", im0)
                  # cv2.waitKey(1)  # 1 millisecond

                  
                  end_time = time.time()  #Calculatio for FPS
                  print(f"The time spent in this loop is: {end_time - start_time}")
                  # fps = 1 / (end_time - start_time)
                  # total_fps += fps
                  # frame_count += 1
                  
                  # fps_list.append(total_fps) #append FPS in list
                  # time_list.append(end_time - start_time) #append time in list
                  

            else: 
              break

        cap.release()
        client_socket.close()
        server_socket.close()
        print("Python script terminated.")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='0', help='video/0 for webcam') #video source
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')   #device arugments
    parser.add_argument('--view-img', action='store_true', help='display results')  #display results
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels') #save confidence in txt writing
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)') #box linethickness
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels') #box hidelabel
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences') #boxhideconf
    opt = parser.parse_args()
    return opt

# #function for plot fps and time comparision graph
# def plot_fps_time_comparision(time_list,fps_list):
#     plt.figure()
#     plt.xlabel('Time (s)')
#     plt.ylabel('FPS')
#     plt.title('FPS and Time Comparision Graph')
#     plt.plot(time_list, fps_list,'b',label="FPS & Time")
#     plt.savefig("FPS_and_Time_Comparision_pose_estimate.png")
    

#main function
def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device,opt.poseweights)
    main(opt)
