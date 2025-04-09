using UnityEngine;
using System;
using System.Diagnostics; // Start Python process
using System.Net.Sockets;
using UnityEngine.UI;
using System.Text;
using System.IO.Enumeration; // Socket communcation
using System.Threading;
using PimDeWitte.UnityMainThreadDispatcher;
using Unity.VisualScripting.Antlr3.Runtime;
using TMPro;
using System.Collections;
using UnityEngine.Networking;
using System.IO;


/**
 * This C# script is for testing the time of current socket communication between Unity and posture detection model.
 * The only difference between this script and the official script is that this one doesnt have process to run the posture detection model.
 * 
 */
public class PlayerMovementWithMVEstimationTest : MonoBehaviour
{
    [SerializeField] private float player_speed = 20;
    [SerializeField] private float default_player_speed = 20; //new
    [SerializeField] private float threshold_player_speed = 40;
    [SerializeField] private float horizontal_speed = 10f;
    [SerializeField] private float jump_force = 60f;

    private int current_path = 1;
    [SerializeField] private float left_path_x = -3.75f;
    [SerializeField] private float middle_path_x = 0f;
    [SerializeField] private float right_path_x = 3.75f;

    private Animator animator;
    private string current_animation;
    private AnimatorStateInfo stateInfo;
    private Vector3 last_position;
    private Vector3 forward_size = Vector3.forward;
    private Rigidbody rb; //for jumping animation
    private PauseInGameMoment pausor;

    private bool isGrounded = true; // Track if the character is grounded
    private bool isSliding = false;
    private bool havingJumpingAnimation = false;
    private bool stopOnCollision = false;
    private bool isSignalReceivedByPoseModel = false;

    private CapsuleCollider capsuleCollider;

    // 0: left, 1: middle, 2: right
    private int hor_detection_result = 1;
    private int future_hor_detection_result = 1;
    // 0: normal, 1: crouch, 2: jump 
    private int vert_detection_result = 0;
    // 0: normal running 1: exercise challenge 2:error phase for missing keypoints, 3:error phase for detecting more than 1 player, 4: Intial keypoint preparation
    private int game_mode = 0;
    private float total_push_up_time = 0; //the time here means the number of push up
    private float total_bridge_time = 0; //the time here means the time (in second) of holding the posture of bridge 
    public bool require_exercise_challenge = false;

    private string loss_keypoints = "!!! Miss Important keypoints !!!";
    private string multipe_players = "!!! Detect multiple players !!!";

    [SerializeField] private RawImage display; // Frame from player
    [SerializeField] private RawImage display_error_phase; // Frame from player in error phase
    [SerializeField] private RawImage display_initial_phase; // Frame from player in intial keypoints preparation phase
    [SerializeField] private GameObject errorPhase;
    [SerializeField] private TextMeshProUGUI error_message;
    [SerializeField] private GameObject initialKeypointPreparationPhase;
    [SerializeField] private AudioSource error_warning_fx;
    [SerializeField] private AudioSource get_ready_audio;
    // For initial phase and error phase
    float audio_repeat_duration = 8f;
    float remaining_time_to_repeat = -1;

    private TcpClient client = null;
    private NetworkStream stream = null;
    private Process python_process = null;
    private Texture2D texture = null;
    private bool connectedToSocket = false;

    private bool isReceivingData = true; // Control flag for the thread

    private string FILE_NAME = Application.streamingAssetsPath + "/venv/Scripts/python";
    private string ARGUMENTS = Application.streamingAssetsPath + "/pose-estimate.py";
    private string WORKING_DIRECTORY = Application.streamingAssetsPath;
    private string ADDRESS = "127.0.0.1";
    private int PORT = 65451; // Port for pose estimation model
    private int READ_TIME_OUT = 5000;


    // For Emotion REcognition
    private string mp3_path = Path.Combine(Application.streamingAssetsPath, "quote.mp3");
    private EmotionRecognition emot_recog_manager = null;
    [SerializeField] private AudioSource quote_audio;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        // assign emotion recongition manager 
        emot_recog_manager = gameObject.GetComponent<EmotionRecognition>();

        animator = GetComponentInChildren<Animator>();

        // assign pasuor to stop game
        GameObject level_control = GameObject.Find("LevelControl");
        pausor = level_control.GetComponent<PauseInGameMoment>();

        AnimatorClipInfo[] animatorinfo = animator.GetCurrentAnimatorClipInfo(0);
        current_animation = animatorinfo[0].clip.name; //get current animation of the character 

        // Assign the Rigidbody component 
        rb = GetComponent<Rigidbody>();

        // Get the Capsule Collider on the Player
        capsuleCollider = GetComponent<CapsuleCollider>();

        // Establish socket socket communication
        StartCoroutine(ConnectToSocket());

        // Use thread to receive movement and frame
        Thread thread = new Thread(GetMovementAndFrame);
        thread.IsBackground = true;
        thread.Start();

        // Freeze the game until we get connect to the models. It will be unfreeze if models are connected in ConnectToSocket
        pausor.PauseInGameState();
    }

    // Update is called once per frame
    void Update()
    {
        //UnityEngine.Debug.Log($"Detection result: {hor_detection_result}");
        //UnityEngine.Debug.Log($"Current path: {current_path}");
        //UnityEngine.Debug.DrawRay(new Vector3(transform.position.x, transform.position.y + bufferCheckDistance, transform.position.z), Vector3.down * bufferCheckDistance, Color.red);
        transform.Translate(Vector3.forward * player_speed * Time.deltaTime, Space.World);
        bool isChanged = IsMovementChange(hor_detection_result, future_hor_detection_result);

        // The reason why using isChanged is that current_path will keep changing 
        if (isChanged)
        {
            hor_detection_result = future_hor_detection_result;
            if ( //move left
                hor_detection_result == 0 &&
                current_path > 0
                )
                current_path--;

            else if ( //move right
                hor_detection_result == 2 &&
                current_path < 2
                )
                current_path++;

            else if (hor_detection_result == 1) //move middle
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

        if (isGrounded && vert_detection_result == 2)
        {
            Jump();
        }
        else if (isGrounded && !isSliding && vert_detection_result == 1)
        {
            Slide();
        }

        //Sometimes OnCollisionEnter is slower to execute again!
        FastAnimationTransition();

        // Use together with shouldRepeatAudio function.
        if(game_mode != 0 || game_mode != 1) remaining_time_to_repeat -= Time.unscaledDeltaTime;

    }
    // See if Movement has changed
    private bool IsMovementChange(int detection_result, int future_detection_result)
    {
        if (detection_result != future_detection_result) return true;
        else return false;
    }

    private void Jump()
    {
        // To prevent double Jump
        havingJumpingAnimation = true; //new
        isGrounded = false;

        // Trigger the "Jump" animation
        ChangeAnimation("Jump");

        // Apply upward force for the jump
        rb.AddForce(Vector3.up * jump_force, ForceMode.Impulse);
    }

    private void Slide()
    {
        isSliding = true;
        // Change CapsuleCollider size while sliding
        Vector3 newCenter = capsuleCollider.center;
        newCenter.y = newCenter.y / 2;
        capsuleCollider.center = newCenter;
        capsuleCollider.height = capsuleCollider.height / 2;

        ChangeAnimation("Running Slide");
    }

    private void ChangeAnimation(string animation, float cross_fade = 0f)
    {
        if (current_animation != animation)
        {
            // When Running Slide animation is finished or is forced to changed to other animation, capsule sollider state need to be changed back.
            if (current_animation == "Running Slide") //new
            {
                Vector3 newCenter = capsuleCollider.center;
                newCenter.y = newCenter.y * 2;
                capsuleCollider.center = newCenter;
                capsuleCollider.height = capsuleCollider.height * 2;

                // While changing animation from sliding, make sure the sliding state is changed also.
                isSliding = false;
            }

            UnityEngine.Debug.Log($"test Changing animation from {current_animation} to {animation}");

            current_animation = animation;



            animator.CrossFade(animation, cross_fade, 0);
        }
    }

    //Detect when the player lands on a ground object.
    //Probably because of the ground mesh, even the character is running on the ground, this method will weirdly periodically execute.
    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.layer == LayerMask.NameToLayer("Ground") && !isSliding && !stopOnCollision) //Only execute the section code while character is not sliding
        {
            //Debug.Log("test Character has landed on the ground.");
            isGrounded = true; // The character is now grounded
        }
    }

    //To deal with the problem that animation stop after finishing Running Slide/Jump and no animation follows
    private void FastAnimationTransition()
    {
        stateInfo = animator.GetCurrentAnimatorStateInfo(0);
        //Debug.Log($"Current animation is: {current_animation}");

        AnimatorClipInfo[] clipInfo = animator.GetCurrentAnimatorClipInfo(0);
        if (clipInfo.Length > 0)
        {
            string currentClipName = clipInfo[0].clip.name;
            //Debug.Log("Current Animation Clip Name: " + currentClipName);
        }

        //Debug.Log("stateInfo.normalizedTime: " + stateInfo.normalizedTime);
        if (stateInfo.IsName("Fast Run") || stateInfo.normalizedTime < 1f) return;

        if (stateInfo.IsName("Running Slide"))
        {
            isSliding = false; //new
            ChangeAnimation("Fast Run");
        }

        // !isSliding is used here, it is because sometimes when character lands on ground and click slide instantly, 
        // it may become Jump -> Slide -> Fast Run (instantly behind Slide), casuing no Slide can be seen. Hence, the varaible is to make sure that character enter sliding animation.

        else if (stateInfo.IsName("Jump") && !isSliding && havingJumpingAnimation) //new
        {
            havingJumpingAnimation = false;
            ChangeAnimation("Fast Run");
        }
    }

    public void StopOnCollision() //new
    {
        stopOnCollision = true;
    }

    public float IncreaseSpeedFromDefault(float percentage) //new
    {

        //Debug.Log($"fatigue increased speed percentage: {percentage}");
        float actual_percentage = percentage / 100; // e.g. 5% -> 0.05
        float result = default_player_speed + default_player_speed * actual_percentage;

        //Debug.Log($"fatigue increased speed actual_percentage: {actual_percentage}");
        //Debug.Log($"fatigue increased speed default_player_speed * actual_percentage: {default_player_speed * actual_percentage}");
        //Debug.Log($"fatigue increased speed result: {result}");

        // Maximum speed should not exceed threshold.
        // If the amount of increment cause the spped to exceed over threshold_player_speed, then reduce the percentage.
        if (result > threshold_player_speed)
        {
            player_speed = threshold_player_speed;
            //Debug.Log($"fatigue increased speed player_speed 111:{player_speed}");
            return percentage -= 5;
        }
        else
        {
            player_speed = result;
            //Debug.Log($"fatigue increased speed player_speed 222:{player_speed}");
            return percentage;
        }

    }

    public void DecreaseSpeedFromDefault(float percentage) //new
    {
        float actual_percentage = percentage / 100; // e.g. 5% -> 0.05
        player_speed = default_player_speed - default_player_speed * actual_percentage;
        UnityEngine.Debug.Log($"fatigue Decrease speed to:{player_speed}");
    }

    public void SetBackToDefaultSpeed() //new
    {
        player_speed = default_player_speed;
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

    IEnumerator ConnectToSocket()
    {
        int maxRetries = 10; // Maximum number of retries
        int retryDelay = 1; // Delay between retries in seconds
        int currentRetry = 0;

        while (currentRetry < maxRetries)
        {
            try
            {
                UnityEngine.Debug.Log($"sunny test Attempting to connect to Python server... (Attempt {currentRetry + 1}/{maxRetries})");
                client = new TcpClient(ADDRESS, PORT);
                stream = client.GetStream();
                stream.ReadTimeout = READ_TIME_OUT;
                UnityEngine.Debug.Log("Connected to Python server");

                //connectedToSocket = true; //EmotionRecognition will set this variable to true
                emot_recog_manager.ConnectedToPoseModel();

                yield break; // Successfully connected, exit coroutine
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogWarning($"sunny test Connection attempt failed: {e.Message}");
                currentRetry++;
            }

            // Move the retry logic outside the catch block
            if (currentRetry < maxRetries)
            {
                // Wait before retrying - this is now outside the catch block
                yield return new WaitForSecondsRealtime(retryDelay);
            }
            else
            {
                UnityEngine.Debug.LogError("sunny test Failed to connect to Python server after multiple attempts.");

#if UNITY_EDITOR
                UnityEditor.EditorApplication.isPlaying = false; // Stops Play Mode
#else
            Application.Quit();
#endif

                yield break;
            }
        }
    }

    private void GetMovementAndFrame()
    {
        while (!connectedToSocket);
        
        bool previousHasPaused = false;
        bool justStartExerciseChallenge = true;

        byte[] game_mode_buffer = new byte[4];
        int bytes_read = 0;
        while (isReceivingData)
        {
            UnityMainThreadDispatcher.Instance().Enqueue(() => UnityEngine.Debug.Log($"game mode: {game_mode}"));
            bytes_read = stream.Read(game_mode_buffer, 0, game_mode_buffer.Length);
            game_mode = BitConverter.ToInt32(game_mode_buffer, 0);

            // Resume from error phase
            if (previousHasPaused && (game_mode != 2 || game_mode !=3))
            {
                UnityMainThreadDispatcher.Instance().Enqueue(() =>
                {
                    pausor.ResumeInGameState();
                    display.enabled = true;
                    errorPhase.SetActive(false);
                    initialKeypointPreparationPhase.SetActive(false);
                });
                previousHasPaused = false;
            }

            if(!justStartExerciseChallenge && game_mode != 1)
            {
                justStartExerciseChallenge = true;
            }

            if (game_mode == 0)
            {
                RunningMode();
                UnityMainThreadDispatcher.Instance().Enqueue(() => StartCoroutine(PlayEmotionRecongitionQuote())); // Play 
            }
            else if(game_mode == 1) //Exercise Challenge mode
            {
                ExerciseChallange(justStartExerciseChallenge);
            }
            else if(game_mode == 2 || game_mode == 3)
            {
                ErrorPhase(!previousHasPaused, game_mode);
                previousHasPaused = true;
            }
            else
            {
                InitialPhase(!previousHasPaused);
                previousHasPaused = true;
            }
        }
    }

    private void RunningMode()
    {
        //DateTime start_time = DateTime.Now;
        try
        {
            // Send exercise challenge request to pose model
            byte[] require_exercise_challenge_buffer = new byte[1];
            require_exercise_challenge_buffer[0] = (byte)(require_exercise_challenge ? 1 : 0);
            stream.Write(require_exercise_challenge_buffer, 0, require_exercise_challenge_buffer.Length);

            // Read detection movement result
            byte[] hor_detection_buffer = new byte[4];
            int bytes_read = stream.Read(hor_detection_buffer, 0, hor_detection_buffer.Length);

            // Handle server disconnection (stream.Read() returning 0)
            if (bytes_read == 0)
            {
                isReceivingData = false;
                DisconnectServer();
            }

            //hor_detection_result = Encoding.UTF8.GetString(detection_buffer).Trim();
            int result = BitConverter.ToInt32(hor_detection_buffer, 0);
            if (result != -1) future_hor_detection_result = result;


            // Read detection movement result
            byte[] vert_detection_buffer = new byte[4];
            bytes_read = stream.Read(vert_detection_buffer, 0, vert_detection_buffer.Length);

            // Handle server disconnection (stream.Read() returning 0)
            if (bytes_read == 0)
            {
                isReceivingData = false; // Stop the thread
                DisconnectServer();
            }

            //vert_detection_result = Encoding.UTF8.GetString(detection_buffer).Trim();
            result = BitConverter.ToInt32(vert_detection_buffer, 0);
            if (result != -1) vert_detection_result = result;
        

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
            //DateTime end_time = DateTime.Now;
            //UnityEngine.Debug.Log($"The time spent in this loop is: {(end_time - start_time).TotalMilliseconds}");
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

    private void ExerciseChallange(bool justStartExerciseChallenge)
    {
        if (justStartExerciseChallenge)
        {
            isSignalReceivedByPoseModel = true;
            justStartExerciseChallenge = false;
            UnityMainThreadDispatcher.Instance().Enqueue(() => UnityEngine.Debug.Log($"player's push_up_time just start exercise:{total_push_up_time}"));
        }

        // Read push up result
        byte[] total_push_up_time_buffer = new byte[4];
        int bytes_read = stream.Read(total_push_up_time_buffer, 0, total_push_up_time_buffer.Length);
        // Handle server disconnection (stream.Read() returning 0)
        if (bytes_read == 0)
        {
            isReceivingData = false;
            DisconnectServer();
        }

        UnityMainThreadDispatcher.Instance().Enqueue(() => UnityEngine.Debug.Log($"player's push_up_time before:{total_push_up_time}"));
        total_push_up_time = BitConverter.ToSingle(total_push_up_time_buffer, 0);

        UnityMainThreadDispatcher.Instance().Enqueue(() => UnityEngine.Debug.Log($"player's push_up_time:{total_push_up_time}"));

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
        });


        // Send the signal to disable exercise challenge request to pose model in an appropriate time
        byte[] require_exercise_challenge_buffer = new byte[1];
        require_exercise_challenge_buffer[0] = (byte)(require_exercise_challenge ? 1 : 0);
        stream.Write(require_exercise_challenge_buffer, 0, require_exercise_challenge_buffer.Length);
    }

    // Used by ExerciseMoment
    public Texture2D ReadTexture()
    {
        return texture;
    }

    // Used by ExerciseMoment
    public void ChangeToExerciseChallenge()
    {
        require_exercise_challenge = true;
    }

    // Used by ExerciseMoment
    public void ChangeFromExerciseChallenge()
    {
        total_push_up_time = 0;
        require_exercise_challenge = false;
        isSignalReceivedByPoseModel = false;
    }

    // Used by ExerciseMoment
    public float GetPushUpTotalTime()
    {
        return total_push_up_time;
    }

    public bool HaveChallengeSignalReceivedByPoseModel()
    {
        return isSignalReceivedByPoseModel;
    }

    public int GetGameMode()
    {
        return game_mode;
    }

    // Used by ExerciseMoment
    private void ErrorPhase(bool isFirstExecution, int game_mode = 2)
    {
        if (isFirstExecution)
        {
            UnityMainThreadDispatcher.Instance().Enqueue(() => {
                pausor.PauseInGameState();
                display.enabled = false;
                errorPhase.SetActive(true);
            });
        }

        if (shouldRepeatAudio()) UnityMainThreadDispatcher.Instance().Enqueue(() => error_warning_fx.Play());

        // Read size of incoming frame
        byte[] size_buffer = new byte[4]; //4 bytes
        stream.Read(size_buffer, 0, size_buffer.Length);
        int frame_size = BitConverter.ToInt32(size_buffer, 0);


        // Read the actual frame
        byte[] frame_buffer = new byte[frame_size];
        int bytes_read = 0;
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
            display_error_phase.texture = texture;
        });

        
        if(game_mode == 2) UnityMainThreadDispatcher.Instance().Enqueue(() => error_message.text = loss_keypoints);
        else UnityMainThreadDispatcher.Instance().Enqueue(() => error_message.text = multipe_players);
    }

    private void InitialPhase(bool isFirstExecution)
    {

        if (isFirstExecution)
        {
            UnityMainThreadDispatcher.Instance().Enqueue(() => {
                pausor.PauseInGameState();
                display.enabled = false;
                initialKeypointPreparationPhase.SetActive(true);
            });
        }

        if (shouldRepeatAudio()) UnityMainThreadDispatcher.Instance().Enqueue(() => get_ready_audio.Play());

        // Read size of incoming frame
        byte[] size_buffer = new byte[4]; //4 bytes
        stream.Read(size_buffer, 0, size_buffer.Length);
        int frame_size = BitConverter.ToInt32(size_buffer, 0);

        // Read the actual frame
        byte[] frame_buffer = new byte[frame_size];
        int bytes_read = 0;
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
            display_initial_phase.texture = texture;
        });
    }

    IEnumerator PlayEmotionRecongitionQuote()
    {
        // Play the audio if we receive signal 
        if (emot_recog_manager.GetSignal() == 1)
        {
            UnityEngine.Debug.Log($"mp3_path:{mp3_path}");
            if (File.Exists(mp3_path))
            {
                string url = "file:///" + mp3_path;
                UnityWebRequest www = UnityWebRequestMultimedia.GetAudioClip(url, AudioType.MPEG);
                yield return www.SendWebRequest();

                if (www.result != UnityWebRequest.Result.Success)
                {
                    UnityEngine.Debug.LogError($"Error loading audio: {www.error}");
                }
                else
                {
                    AudioClip my_quote = DownloadHandlerAudioClip.GetContent(www);
                    quote_audio.clip = my_quote;
                    quote_audio.Play();
                }

            }
            else
            {
                UnityEngine.Debug.LogError("quote mp3 does not exist in the path");
            }

            emot_recog_manager.ResetSignal();
        }
    }

    private bool shouldRepeatAudio()
    {
        //UnityEngine.Debug.Log($"remaining_time_to_repeat: {remaining_time_to_repeat}");
        if (remaining_time_to_repeat < 0)
        {
            remaining_time_to_repeat = audio_repeat_duration;
            return true;
        }
        else return false;
    }

    public void StopThread()
    {
        isReceivingData = false;
    }

    public void SetConnectedToSocket(bool connected)
    {
        connectedToSocket = connected;
    }

    private void DisconnectServer()
    {
        UnityMainThreadDispatcher.Instance().Enqueue(() =>
        {
            UnityEngine.Debug.LogError("Server closed the connection or connection lost.");
            #if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false; // Stops Play Mode
            #else
            Application.Quit();
            #endif
        });
    }
}
