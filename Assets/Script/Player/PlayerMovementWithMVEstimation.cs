using UnityEngine;
using System;
using System.Diagnostics; // Start Python process
using System.Net.Sockets;
using UnityEngine.UI;
using System.Text;
using System.IO.Enumeration; // Socket communcation
using System.Threading;
using PimDeWitte.UnityMainThreadDispatcher;


public class PlayerMovementWithMVEstimation : MonoBehaviour
{
    [SerializeField] private float player_speed = 10;
    [SerializeField] private float horizontal_speed = 10f;
    [SerializeField] private float jump_force = 60f;
    [SerializeField] private float bufferCheckDistance = 0.1f;

    private int current_path = 1;
    [SerializeField] private float left_path_x = -3.75f;
    [SerializeField] private float middle_path_x = 0f;
    [SerializeField] private float right_path_x = 3.75f;
    RaycastHit hit;

    private Animator animator;
    private string current_animation;
    private AnimatorStateInfo stateInfo;

    //for jumping animation
    private Rigidbody rb;

    private bool isGrounded = false; // Track if the character is grounded
    private bool isSliding = false;

    private CapsuleCollider capsuleCollider;

    // 0: left, 1: middle, 2: right
    int detection_result = 1;
    int future_detection_result = 1;
    public RawImage display; // Image from player
    private TcpClient client = null;
    private NetworkStream stream = null;
    private Process python_process = null;
    [SerializeField] private Texture2D texture = null;

    private bool isReceivingData = true; // Control flag for the thread

    private string FILE_NAME = Application.streamingAssetsPath + "/venv/Scripts/python";
    private string ARGUMENTS = Application.streamingAssetsPath + "/pose-estimate.py";
    private string WORKING_DIRECTORY = Application.streamingAssetsPath;
    private string ADDRESS = "127.0.0.1";
    private int PORT = 65451;
    private int READ_TIME_OUT = 5000;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        UnityEngine.Debug.Log("test1");
        animator = GetComponentInChildren<Animator>();
        //runningSlideDuration = getRunningSlideAnimationDuration();

        AnimatorClipInfo[] animatorinfo = animator.GetCurrentAnimatorClipInfo(0);
        current_animation = animatorinfo[0].clip.name; //get current animation of the character 

        // Assign the Rigidbody component 
        rb = GetComponent<Rigidbody>();

        // Get the Capsule Collider on the Player
        capsuleCollider = GetComponent<CapsuleCollider>();
        UnityEngine.Debug.Log("test2");

        // Start python script 
        StartPythonScript();
        // Establish socket socket communication
        ConnectToPythonServer();

        // Use thread to receive movement and frame
        Thread thread = new Thread(GetMovementAndFrame);
        thread.IsBackground = true;
        thread.Start();
    }

    // Update is called once per frame
    void Update()
    {
        UnityEngine.Debug.Log($"Detection result: {detection_result}");
        UnityEngine.Debug.Log($"Current path: {current_path}");
        UnityEngine.Debug.DrawRay(new Vector3(transform.position.x, transform.position.y + bufferCheckDistance, transform.position.z), Vector3.down * bufferCheckDistance, Color.red);
        transform.Translate(Vector3.forward * player_speed * Time.deltaTime, Space.World);
        bool isChanged = IsMovementChange(detection_result, future_detection_result);

        if (isChanged)
        {
            detection_result = future_detection_result;
            if ( //move left
                detection_result == 0 &&
                current_path > 0
                )
                current_path--;

            else if ( //move right
                detection_result == 2 &&
                current_path < 2
                )
                current_path++;

            else if (detection_result == 1) //move middle
                current_path = 1;
        }

        float target_x = middle_path_x;
        if (current_path == 0)
        {
            target_x = left_path_x;
        }
        else if (current_path == 2)
        {
            target_x = right_path_x;
        }
        Vector3 target_pos = new Vector3(target_x, transform.position.y, transform.position.z);
        transform.position = Vector3.Lerp(transform.position, target_pos, horizontal_speed * Time.deltaTime);

        if (isGrounded && Input.GetKeyDown(KeyCode.Space))
        {
            Jump();
        }
        else if (isGrounded && Input.GetKeyDown(KeyCode.LeftControl))
        {
            Slide();
        }

        //Sometimes OnCollisionEnter is slower to execute again!
        FasterAnimationTransition();

    }
    // See if Movement has changed
    private bool IsMovementChange(int detection_result, int future_detection_result)
    {
        if (detection_result != future_detection_result) return true;
        else return false;
    }

    private void Jump()
    {
        // Apply upward force for the jump
        rb.AddForce(Vector3.up * jump_force, ForceMode.Impulse);

        // Trigger the "Jump" animation
        ChangeAnimation("Jump");
    }

    private void Slide()
    {
        isSliding = true;

        Vector3 newCenter = capsuleCollider.center;
        newCenter.y = newCenter.y / 2;
        capsuleCollider.center = newCenter;
        capsuleCollider.height = capsuleCollider.height / 2;

        ChangeAnimation("Running Slide");
    }

    private void ChangeAnimation(string animation, float cross_fade = 0.2f)
    {
        if(current_animation != animation)
        {
            //Debug.Log($"Changing animation from {current_animation} to {animation}");
            current_animation = animation;
            animator.CrossFade(animation, cross_fade, 0);
        }
    }
    //Detect when the player lands on a ground object
    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.layer == LayerMask.NameToLayer("Ground") && !isSliding) //Only execute the section code while character is not sliding
       {
            UnityEngine.Debug.Log("Character has landed on the ground.");
            isGrounded = true; // The character is now grounded
            ChangeAnimation("Fast Run");
        }
        //Jump character is sliding and Jump, there will be no problem to execute the following code
        else if (isSliding)
        {
            UnityEngine.Debug.Log("Is sliding");

            stateInfo = animator.GetCurrentAnimatorStateInfo(0);
            if (stateInfo.normalizedTime >= 1f)
            {
                isSliding = false;
                Vector3 newCenter = capsuleCollider.center;
                newCenter.y = newCenter.y * 2;
                capsuleCollider.center = newCenter;
                capsuleCollider.height = capsuleCollider.height * 2;
            }
        }
    }

    //Ensure the character stays grounded while on the ground object
    private void OnCollisionStay(Collision collision)
    {
        if (collision.gameObject.layer == LayerMask.NameToLayer("Ground"))
        {
            isGrounded = true; // The character is still grounded
        }
    }

    // Detect when the player leaves the ground object
    private void OnCollisionExit(Collision collision)
    {
        if (collision.gameObject.layer == LayerMask.NameToLayer("Ground"))
        {
            //Debug.Log("Character has left the ground.");
            isGrounded = false; // The character is no longer grounded
        }
    }

    //To deal with the problem that animation stop after finishing Running Slide/Jump and no animation follows
    private void FasterAnimationTransition()
    {
        stateInfo = animator.GetCurrentAnimatorStateInfo(0);
        if (stateInfo.IsName("Fast Run") || stateInfo.normalizedTime < 1f) return;

        if (stateInfo.IsName("Running Slide"))
        {
            ChangeAnimation("Fast Run");
        }
        else if (stateInfo.IsName("Jump")) ChangeAnimation("Fast Run");
    }

    private void OnApplicationQuit()
    {
        // Stop the thread
        isReceivingData = false;

        try
        {
            if(stream != null)
            {
                byte[] exit_message = Encoding.UTF8.GetBytes("exit");
                stream.Write(exit_message, 0, exit_message.Length);
                stream.Flush();
            }
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogError($"Error while sending termination signal: {e.Message}");
        }
        if(stream != null) stream.Close();
        if(client != null) client.Close();

        if(python_process != null && python_process.HasExited)
        {
            python_process.Kill();
            python_process.Dispose();
        }
    }

    void StartPythonScript()
    {
        try
        {
            ProcessStartInfo start = new ProcessStartInfo
            {
                FileName = FILE_NAME,
                Arguments = ARGUMENTS,
                WorkingDirectory = WORKING_DIRECTORY,
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            python_process = Process.Start(start);

            // Read the error streams asynchronously (OutputDataReceived and ErrorDataReceived is event, lambda function is event handler)
            python_process.ErrorDataReceived += (sender, args) =>
            {
                if (!string.IsNullOrEmpty(args.Data))
                    UnityEngine.Debug.LogError($"Python Error: {args.Data}");
            };
            python_process.BeginErrorReadLine();
            UnityEngine.Debug.Log("Python script started.");
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogError($"Error in creating pythin script process: {e.Message}");
            #if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false; // Stops Play Mode
            #else
            Application.Quit();
            #endif
        }
    }

    private void ConnectToPythonServer()
    {
        int maxRetries = 10; // Maximum number of retries
        int retryDelay = 1000; // Delay between retries in milliseconds (1 second)
        int currentRetry = 0;

        while (currentRetry < maxRetries)
        {
            try
            {
                UnityEngine.Debug.Log($"Attempting to connect to Python server... (Attempt {currentRetry + 1}/{maxRetries})");
                client = new TcpClient(ADDRESS, PORT);
                stream = client.GetStream();
                stream.ReadTimeout = READ_TIME_OUT;
                UnityEngine.Debug.Log("Connected to Python server");
                return; // Exit the function if connection is successful
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogWarning($"Connection attempt failed: {e.Message}");
                currentRetry++;
                if (currentRetry < maxRetries)
                {
                    System.Threading.Thread.Sleep(retryDelay); // Wait before retrying
                }
                else
                {
                    UnityEngine.Debug.LogError("Failed to connect to Python server after multiple attempts.");
                    #if UNITY_EDITOR
                    UnityEditor.EditorApplication.isPlaying = false; // Stops Play Mode
                    #else
                    Application.Quit();
                    #endif
                }
            }
        }
    }

    private void GetMovementAndFrame()
    {
        while (isReceivingData)
        {
            try
            {
                // Read detection movement result
                byte[] detection_buffer = new byte[4];
                int bytes_read = stream.Read(detection_buffer, 0, detection_buffer.Length);

                // Handle server disconnection (stream.Read() returning 0)
                if (bytes_read == 0)
                {
                    UnityMainThreadDispatcher.Instance().Enqueue(() =>
                    {
                        UnityEngine.Debug.LogError("Server closed the connection or connection lost.");
                        isReceivingData = false; // Stop the thread
                    #if UNITY_EDITOR
                        UnityEditor.EditorApplication.isPlaying = false; // Stops Play Mode
                    #else
                        Application.Quit();
                    #endif
                    });
                    break; // Exit the loop
                }


                UnityMainThreadDispatcher.Instance().Enqueue(() =>{
                    UnityEngine.Debug.Log($"First MainThreadDispatcher in GetMovementAndFrame");
                    //detection_result = Encoding.UTF8.GetString(detection_buffer).Trim();
                    int result = BitConverter.ToInt32(detection_buffer, 0);
                    if (result != -1) future_detection_result = result;
                });

                // Read size of incoming frame
                byte[] size_buffer = new byte[4]; //4 bytes
                stream.Read(size_buffer, 0, size_buffer.Length);
                int frame_size = BitConverter.ToInt32(size_buffer, 0);


                // Read the actual frame
                byte[] frame_buffer = new byte[frame_size];
                bytes_read = 0;
                while (bytes_read < frame_size)
                {
                    bytes_read += stream.Read(frame_buffer, bytes_read, frame_size - bytes_read);
                }
                UnityMainThreadDispatcher.Instance().Enqueue(() =>
                {
                    if (texture == null)
                    {
                        texture = new Texture2D(2, 2); //PLaceholder sizem will resize automatically 
                    }
                    texture.LoadImage(frame_buffer); //Decode JPEG data
                    display.texture = texture;
                });
            }
            catch (Exception e)
            {
                UnityMainThreadDispatcher.Instance().Enqueue(() => {
                    isReceivingData = false;
                    UnityEngine.Debug.LogError($"Error receiving data: {e.Message}");
                    #if UNITY_EDITOR
                    UnityEditor.EditorApplication.isPlaying = false; // Stops Play Mode
                    #else
                    Application.Quit();
                    #endif
                });
            }
        }
    }
}
