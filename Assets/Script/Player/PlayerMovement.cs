using UnityEngine;
using System.Collections;
using System;
using UnityEditor;

public class PlayerMovement : MonoBehaviour
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

    //for jumping animation
    private Rigidbody rb;

    private bool isGrounded = true; // Track if the character is grounded
    private bool isSliding = false;
    private bool havingJumpingAnimation = false;
    private bool stopOnCollision = false;

    private CapsuleCollider capsuleCollider;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        default_player_speed = player_speed;

        animator = GetComponentInChildren<Animator>();

        //runningSlideDuration = getRunningSlideAnimationDuration();

        AnimatorClipInfo[] animatorinfo = animator.GetCurrentAnimatorClipInfo(0);
        current_animation = animatorinfo[0].clip.name; //get current animation of the character 

        // Assign the Rigidbody component 
        rb = GetComponent<Rigidbody>();

        // Prevent tunneling while character is moving too fast
        rb.collisionDetectionMode = CollisionDetectionMode.Continuous;

        // Get the Capsule Collider on the Player
        capsuleCollider = GetComponent<CapsuleCollider>();

    }

    // Update is called once per frame
    void Update()
    {
        last_position = transform.position;
        transform.Translate(forward_size * player_speed * Time.deltaTime, Space.World);

        if ( //move left
            Input.GetKey(KeyCode.A) &&
            current_path > 0
            )
        {
            current_path--;
        }
        else if ( //move right
            Input.GetKey(KeyCode.D) &&
            current_path < 2
            )
        {
            current_path++;
        }
        else if (Input.GetKeyDown(KeyCode.S))
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
        // Move direction (i.e. move to different path)
        transform.position = Vector3.Lerp(transform.position, target_pos, horizontal_speed * Time.deltaTime);


        if (isGrounded && Input.GetKeyDown(KeyCode.Space)) //new
        {
            Jump();
        }
        else if (isGrounded && !isSliding && Input.GetKeyDown(KeyCode.LeftControl))
        {
            Slide();
        }

        FastAnimationTransition();
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
            if(current_animation == "Running Slide") //new
            {
                Vector3 newCenter = capsuleCollider.center;
                newCenter.y = newCenter.y * 2;
                capsuleCollider.center = newCenter;
                capsuleCollider.height = capsuleCollider.height * 2;

                // While changing animation from sliding, make sure the sliding state is changed also.
                isSliding = false;
            }

            Debug.Log($"test Changing animation from {current_animation} to {animation}");

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
        Debug.Log($"fatigue Decrease speed to:{player_speed}");
    }

    public void SetBackToDefaultSpeed() //new
    {
        player_speed = default_player_speed;
    }
}
