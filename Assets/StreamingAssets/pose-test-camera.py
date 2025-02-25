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
        # server_socket = socket(AF_INET, SOCK_STREAM) # Sunny
        # server_socket.bind((HOST, PORT))
        # server_socket.listen(1)
        # print("Waiting for Unity to connect...")
        # ret, frame = cap.read() # Let camera operate before real task execution
        # client_socket, addr = server_socket.accept() # Accept connect when cap is opened
        # print(f"Connected to Unity at {addr}") # Sunny

        # threading.Thread(target=listen_for_exit_signal, args=(client_socket,), daemon = True) # Sunny

        frame_width = int(cap.get(3)) #get video frame width
        if frame_width > 800: # Sunny
          frame_width = 800

        while(cap.isOpened): #loop until cap opened or video not complete
            #print("Frame {} Processing".format(frame_count+1)) # Sunny

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
                start_time = time.time() #start time for fps calculation
            
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
            
                cv2.imshow("YOLOv7 Pose Estimation Demo", im0)
                cv2.waitKey(1)  # 1 millisecond

                
                end_time = time.time()  #Calculatio for FPS
                print(f"The time spent in this loop is: {end_time - start_time}")


            else: 
              break

        cap.release()
        # client_socket.close()
        # server_socket.close()
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
