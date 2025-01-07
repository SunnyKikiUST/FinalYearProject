using UnityEngine;
using System.Collections;
using System;
using UnityEditor;

public class PlayerMovement : MonoBehaviour
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

    }

    // Update is called once per frame
    void Update()
    {
        Debug.DrawRay(new Vector3(transform.position.x, transform.position.y + bufferCheckDistance, transform.position.z), Vector3.down * bufferCheckDistance, Color.red);
        transform.Translate(Vector3.forward * player_speed * Time.deltaTime, Space.World);

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
            Debug.Log("Character has landed on the ground.");
            isGrounded = true; // The character is now grounded
            ChangeAnimation("Fast Run");
        }
        //Jump character is sliding and Jump, there will be no problem to execute the following code
        else if (isSliding)
        {
            Debug.Log("Is sliding");

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
}
