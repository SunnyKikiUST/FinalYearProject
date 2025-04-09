import os
import platform

COMPANY = 'HKUST'
PRODUCT = 'FitnessFugitive'
FILE_PATH = 'settings.txt'

# Function for receiving termination signal for script
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

def get_persistent_data_path(company=COMPANY, product=PRODUCT):
    system = platform.system()

    if system == "Windows":
        # On Windows, Unity uses the LocalLow folder under the user's AppData folder.
        return os.path.join(os.path.expanduser("~"), "AppData", "LocalLow", company, product)
    else:
        raise NotImplementedError(f"Unsupported platform: {system}")

# Get openai key from setting.txt
def get_setting_openAI_key():
    directory = get_persistent_data_path()
    file_path = os.path.join(directory, "settings.txt")
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            # Read all lines and strip newline characters
            lines = [line.strip() for line in f.readlines()]

        openai_key = lines[0] if len(lines) > 0 else None

        return openai_key
    else:
        print("Settings file not found.")
        return None

# Get webcam_index from setting.txt
def get_setting_webcam_index():
    directory = get_persistent_data_path()
    file_path = os.path.join(directory, "settings.txt")
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            # Read all lines and strip newline characters
            lines = [line.strip() for line in f.readlines()]

        webcam_index = lines[1] if len(lines) > 0 else None

        return webcam_index
    else:
        print("Settings file not found.")
        return None
