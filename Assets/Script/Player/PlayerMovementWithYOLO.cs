using UnityEngine;
using System;
using System.Diagnostics; // Start Python process
using System.Net.Sockets;
using UnityEngine.UI;
using System.Text; // Socket communcation


public class PlayerMovementWithYOLO : MonoBehaviour
{
    [SerializeField] private float player_speed = 20;
    [SerializeField] private float horizontal_speed = 1.5f;
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

    string detection_result = "middle";
    public RawImage display; // Image from player
    private TcpClient client;
    private NetworkStream stream;
    private Process python_process;
    private Texture2D texture;


    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        animator = GetComponentInChildren<Animator>();
        //runningSlideDuration = getRunningSlideAnimationDuration();

        AnimatorClipInfo[] animatorinfo = animator.GetCurrentAnimatorClipInfo(0);
        current_animation = animatorinfo[0].clip.name; //get current animation of the character 

        // Assign the Rigidbody component 
        rb = GetComponent<Rigidbody>();

        // Get the Capsule Collider on the Player
        capsuleCollider = GetComponent<CapsuleCollider>();

        StartPythonScript();
        ConnectToPythonServer();
    }

    // Update is called once per frame
    void Update()
    {
        UnityEngine.Debug.DrawRay(new Vector3(transform.position.x, transform.position.y + bufferCheckDistance, transform.position.z), Vector3.down * bufferCheckDistance, Color.red);
        transform.Translate(Vector3.forward * player_speed * Time.deltaTime, Space.World);

        if ( //move left
            detection_result == "left" &&
            current_path > 0
            ) 
        {
            current_path--;
        }
        else if ( //move right
            detection_result == "middle" &&
            current_path < 2
            ) 
        {
            current_path++;
        }
        else if (detection_result == "right")
        {
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

            // Reactivate collision processing after the cooldown (equivalent for waiting sliding animation finish)
            //StartCoroutine(EnableCollisionAfterCooldown());

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

    //private float getRunningSlideAnimationDuration()
    //{
        // Get all the animation clips in the Animator
    //    AnimationClip[] clips = animator.runtimeAnimatorController.animationClips;

        // Find the animation clip named "RunningSlide"
    //    AnimationClip runningSlideClip = Array.Find(clips, clip => clip.name == "RunningSlide");

    //     if (runningSlideClip != null) return runningSlideClip.length;
    //    else return -1f;
    //}

    //private IEnumerator EnableCollisionAfterCooldown() 
    //{
    //    yield return new WaitForSeconds(runningSlideDuration - durationMinimize);
    //    isSliding = false;

    //    Vector3 newCenter = capsuleCollider.center;
    //    newCenter.y = newCenter.y * 2;
    //    capsuleCollider.center = newCenter;
    //    capsuleCollider.height = capsuleCollider.height * 2;

    //    Debug.Log("Collision processing re-enabled.");
    //}

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
        if(stream != null) stream.Close();
        if(client != null) client.Close();
    }

    void StartPythonScript()
    {
        python_process = new Process();


        UnityEngine.Debug.Log("Python script started.");
    }

    private void ConnectToPythonServer()
    {
        try
        {
            client = new TcpClient("127.0.0.1", 65432);
            stream = client.GetStream();
            UnityEngine.Debug.Log("Connected to Python server");
        }
        catch(Exception e)
        {
            UnityEngine.Debug.LogError($"Socket connection error:{e.Message}");
        }
    }

    private void GetMovementAndPhoto()
    {
        try
        {
            // Read detection movement result
            byte[] detection_buffer = new byte[10];
            stream.Read(detection_buffer, 0, detection_buffer.Length);
            detection_result = Encoding.UTF8.GetString(detection_buffer).Trim();
            UnityEngine.Debug.Log($"Detection result: {detection_result}");

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

            if (texture == null)
            {
                texture = new Texture2D(2, 2); //PLaceholder sizem will resize automatically 
            }
            texture.LoadImage(frame_buffer); //Decode JPEG data
            display.texture = texture;
        }
        catch(Exception e)
        {
            UnityEngine.Debug.LogError($"Error receiving data: {e.Message}");
        }
    }
}
