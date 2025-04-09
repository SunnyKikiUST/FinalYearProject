import cv2
import threading
import time
import numpy as np
import argparse
import sys
import multiprocessing
import os
from multiprocessing import shared_memory, Process, Queue

# Add a global error queue to communicate errors between processes
error_queue = Queue()

def run_pose_script(error_queue):
    """Run the pose estimation script as a separate process"""
    try:
        import importlib.util
        import sys
        import os
        import argparse
        
        # Get the absolute path to the script
        #script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose-estimate_updated_withsocket.py")
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose-estimate_updated.py")

        try:
            # Create a module spec
            spec = importlib.util.spec_from_file_location("pose_estimate", script_path) #A module specification (or spec) describes how a module is to be imported and loaded.
            pose_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(pose_module)
        

            # Execute the main function of the module
            pose_module.process_main()  # If this calls sys.exit(0), it will raise SystemExit.
            print("come back from main function")
            error_queue.put("Pose module completed successfully.")
                
        except BaseException  as e:
            print(f"Error running pose estimation module: {e}")
            error_queue.put(f"Pose module error: {str(e)}")
            
    except BaseException as e:
        print(f"Error in pose estimation process: {e}")
        error_queue.put(f"Pose module error: {str(e)}")

def run_facial_recognition_script():
    """Run the pose estimation script as a separate process"""
    try:
        import importlib.util
        import sys
        import os
        import argparse
        
        # Get the absolute path to the script
        #script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose-estimate_updated_withsocket.py")
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "facial_recognition_with_gpt.py")

        try:
            # Create a module spec
            spec = importlib.util.spec_from_file_location("facial_recognition_with_gpt", script_path) #A module specification (or spec) describes how a module is to be imported and loaded.
            pose_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(pose_module)
        
            pose_module.process_main() # Use the function "process_main " defined in the module
            
        except Exception as e:
            print(f"Error running facial expression recognition module: {e}")
            error_queue.put(f"Facial expression recognition module error: {str(e)}")
            
    except Exception as e:
        print(f"Error in acial expression recognition process: {e}")
        error_queue.put(f"Pose module error: {str(e)}")

class CameraStreamer:
    def __init__(self, camera_source=0, 
                 low_res=(640, 480),
                 high_res=(1280, 960),
                 fallback_high_res=(960, 720)):
        self.camera_source = camera_source
        self.low_res = low_res
        self.high_res = high_res
        self.fallback_high_res = fallback_high_res
        
        self.running = False
        self.cap = None
        
        # Shared memory names
        self.low_res_shm_name = "fitness_fat_camera_low_res"
        self.high_res_shm_name = "fitness_fat_camera_high_res"
        
        # Shared memory objects
        self.low_res_shm = None
        self.high_res_shm = None
        
        # Pose estimation process
        self.pose_process = None
        
    def initialize_camera(self):
        """Initialize camera with highest supported resolution"""
        self.cap = cv2.VideoCapture(self.camera_source)
        
        # Try to set high resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.high_res[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.high_res[1])
        
        # Check if the camera supports the requested high resolution
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        print(f"Camera initialized with resolution: {actual_width}x{actual_height}")
        
        # If not supported, try fallback resolution
        if actual_width < self.high_res[0] or actual_height < self.high_res[1]:
            print(f"Camera doesn't support {self.high_res[0]}x{self.high_res[1]}, trying fallback: {self.fallback_high_res[0]}x{self.fallback_high_res[1]}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.fallback_high_res[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.fallback_high_res[1])
            
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"Camera set to: {actual_width}x{actual_height}")
            
            # Update high_res to actual values
            self.high_res = (int(actual_width), int(actual_height))
        
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            error_queue.put("Failed to open camera")
            return False
            
        return True
    
    def initialize_shared_memory(self):
        """Initialize shared memory for both resolutions"""
        # Calculate buffer sizes
        low_res_size = self.low_res[0] * self.low_res[1] * 3  # RGB format
        high_res_size = self.high_res[0] * self.high_res[1] * 3  # RGB format
        
        try:
            # Clean up any existing shared memory with the same name
            try:
                temp = shared_memory.SharedMemory(name=self.low_res_shm_name, create=False)
                temp.close()
                temp.unlink()
            except FileNotFoundError:
                pass
            
            try:
                temp = shared_memory.SharedMemory(name=self.high_res_shm_name, create=False)
                temp.close()
                temp.unlink()
            except FileNotFoundError:
                pass
            
            # Create new shared memory segments
            self.low_res_shm = shared_memory.SharedMemory(
                name=self.low_res_shm_name, create=True, size=low_res_size)
            
            self.high_res_shm = shared_memory.SharedMemory(
                name=self.high_res_shm_name, create=True, size=high_res_size)
            
            print(f"Initialized shared memory segments:")
            print(f"  - Low resolution ({self.low_res[0]}x{self.low_res[1]}): {self.low_res_shm_name}")
            print(f"  - High resolution ({self.high_res[0]}x{self.high_res[1]}): {self.high_res_shm_name}")
            
            return True
            
        except Exception as e:
            print(f"Error initializing shared memory: {e}")
            error_queue.put(f"Shared memory error: {str(e)}")
            self.clean_shared_memory()
            return False
            
    def update_shared_memory(self, frame, is_low_res=True):
        """Update the shared memory with a new frame"""
        try:
            # Convert frame to contiguous array and copy to shared memory
            frame_bytes = np.ascontiguousarray(frame).tobytes()
            
            if is_low_res:
                self.low_res_shm.buf[:len(frame_bytes)] = frame_bytes
            else:
                self.high_res_shm.buf[:len(frame_bytes)] = frame_bytes
                
        except Exception as e:
            print(f"Error updating shared memory: {e}")
            error_queue.put(f"Failed to update shared memory: {str(e)}")
    
    def start_pose_estimation(self):
        """Start pose estimation process"""
        try:
            print("Start pose estimation")
            # Start pose estimation in a separate process using the global function
            self.pose_process = Process(target=run_pose_script, args=(error_queue,))
            self.pose_process.daemon = True
            self.pose_process.start()
            
            print("Pose estimation process started with PID:", self.pose_process.pid)
            
        except Exception as e:
            print(f"Error starting pose estimation: {e}")
            error_queue.put(f"Failed to start pose process: {str(e)}")
    
    def stream(self):
        """Capture and stream camera frames to shared memory"""
        self.running = True
        
        if not self.initialize_camera():
            return
            
        if not self.initialize_shared_memory():
            return
        
        try:
            frame_count = 0
            start_time = time.time()
            
            while self.running:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Error: Failed to capture frame.")
                    error_queue.put("Failed to capture frame")
                    break
                
                # Create resized frames for low resolution
                low_res_frame = cv2.resize(frame, self.low_res)

                low_res_frame = cv2.cvtColor(low_res_frame, cv2.COLOR_BGR2RGB)
                
                # High-res frame might be the original if camera already matches
                if frame.shape[1] != self.high_res[0] or frame.shape[0] != self.high_res[1]:
                    high_res_frame = cv2.resize(frame, self.high_res)
                else:
                    high_res_frame = frame

                high_res_frame = cv2.cvtColor(high_res_frame, cv2.COLOR_BGR2RGB)
                
                # Update shared memory with new frames
                self.update_shared_memory(low_res_frame, is_low_res=True)
                self.update_shared_memory(high_res_frame, is_low_res=False)
                
                # Calculate FPS every 30 frames
                # frame_count += 1
                # if frame_count % 30 == 0:
                #     end_time = time.time()
                #     fps = 30 / (end_time - start_time)
                #     print(f"Camera FPS: {fps:.2f}")
                #     start_time = time.time()
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("Streaming stopped by user.")
        except Exception as e:
            print(f"Error in stream: {e}")
            error_queue.put(f"Stream error: {str(e)}")
        finally:
            self.stop()
    
    def clean_shared_memory(self):
        """Clean up shared memory resources"""
        try:
            if self.low_res_shm is not None:
                self.low_res_shm.close()
                self.low_res_shm.unlink()
                
            if self.high_res_shm is not None:
                self.high_res_shm.close()
                self.high_res_shm.unlink()
                
        except Exception as e:
            print(f"Error cleaning shared memory: {e}")
    
    def stop_pose_estimation(self):
        """Stop the pose estimation process"""
        if self.pose_process is not None and self.pose_process.is_alive():
            print("Terminating pose estimation process...")
            self.pose_process.terminate()
            self.pose_process.join(timeout=2)
            
            if self.pose_process.is_alive():
                print("Force killing pose estimation process...")
                self.pose_process.kill()
            
            print("Pose estimation process stopped.")
    
    def stop(self):
        """Stop streaming and release resources"""
        self.running = False
        
        self.stop_pose_estimation()
        
        # Release camera resources
        if self.cap is not None:
            self.cap.release()
        
        # Clean up shared memory
        self.clean_shared_memory()
        
        cv2.destroyAllWindows()
        print("Camera streaming stopped.")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Stream camera frames using shared memory')
    parser.add_argument('--source', type=int, default=0, help='Camera source index')
    parser.add_argument('--with-pose', action='store_true', default=True, help='Start pose estimation alongside camera streaming')
    parser.add_argument('--device', type=str, default='cpu', help='Device for pose estimation (cpu/cuda)')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    streamer = CameraStreamer(
        camera_source=args.source
    )
    
    print(f"Starting camera stream with shared memory:")
    print(f"  - Low resolution (640x480)")
    print(f"  - High resolution (1280x960 or 960x720)")
    
    # Start streaming in a new thread
    stream_thread = threading.Thread(target=streamer.stream)
    stream_thread.daemon = True
    stream_thread.start()
    
    # Start pose estimation if requested (after shared memory is initialized)
    if args.with_pose:
        # Wait a bit for shared memory to be initialized
        time.sleep(2)
        streamer.start_pose_estimation()
    
    try:
        print("going to check stream_thread.is_alive():")
        while stream_thread.is_alive():
            # Check for errors in the queue
            print("inside while loop")
            if not error_queue.empty():
                error_msg = error_queue.get()
                print(f"Critical error detected: {error_msg}")
                print("Stopping application...")
                streamer.stop()
                stream_thread.join(timeout=2)
                sys.exit(1)  # Exit with error code
            time.sleep(3)
        print("stream_thread is not alive anymore")
    except KeyboardInterrupt:
        print("Stopping...")
        streamer.stop()
        stream_thread.join(timeout=2)
        return

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()