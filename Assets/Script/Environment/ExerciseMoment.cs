using TMPro;
using Unity.VisualScripting.Antlr3.Runtime;
using UnityEngine;
using System;
using UnityEngine.UI;
using System.Collections;


// This should be attached to Gameobject "ExerciseChallengePlane"
public class ExerciseMoment : MonoBehaviour
{
    [SerializeField] private GameObject player;
    [SerializeField] private PlayerMovementWithMVEstimationTest player_movement;
    [SerializeField] private GameObject ExerciseChallenge;

    [SerializeField] private GameObject pushUpUI;
    [SerializeField] private GameObject bridgeUI;

    // Represent the camera streaming in non exercise challenge in game. Have a raw image that is attached to the game object.  
    [SerializeField] private GameObject ingameRealTimeStreaming;
    [SerializeField] private RawImage pushup_real_time_streaming;
    [SerializeField] private RawImage bridge_real_time_streaming;

    // the count here mean total push up time that has done
    [SerializeField] private TextMeshProUGUI pushUpUI_count;
    [SerializeField] private TextMeshProUGUI pushUpUI_timeleft;

    // the count here mean total accumulated time
    [SerializeField] private TextMeshProUGUI bridgeUI_count;
    [SerializeField] private TextMeshProUGUI bridgeUI_timeleft;

    // For restarting counting
    [SerializeField] private GameObject gameRestartCountingExercise;
    [SerializeField] private GameObject count_down_3;
    [SerializeField] private GameObject count_down_2;
    [SerializeField] private GameObject count_down_1;
    [SerializeField] private GameObject count_down_exercise;
    [SerializeField] private AudioSource count_down_sound_effect;
    [SerializeField] private AudioSource game_start_sound_effect;
    [SerializeField] private AudioSource exercise_fail;
    [SerializeField] private AudioSource exercise_succeed;
    [SerializeField] private GameObject success_image;
    [SerializeField] private GameObject fail_image;

    // Exercise mode fx
    [SerializeField] private AudioSource one_succeed;
    [SerializeField] private AudioSource time_warning;

    [SerializeField] private int push_up_require = 2;
    [SerializeField] private int bridge_require = 5;


    private FatigueMonitoringSystem fatigueMonitoringSystem;


    private PauseInGameMoment pausor;
    private bool isTriggered = false;
    private bool inExerciseChallenge = false;
    private bool exChallengeSignalReceivedByPoseModel = false;

    private int test_randowmDetermination = 0;

    private float push_up_time = 0; // Number of time to do push up
    private float bridge_time = 0; // Total time (in second) for holding the bridge posture
    private int remaining_warning_time = 3;

    [SerializeField] private float remaining_push_up_time = 8f;
    [SerializeField] private float remaining_bridge_time = 20f;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        GameObject level_control = GameObject.Find("LevelControl");
        pausor = level_control.GetComponent<PauseInGameMoment>();
        player_movement = player.GetComponent<PlayerMovementWithMVEstimationTest>();
        fatigueMonitoringSystem = level_control.GetComponent<FatigueMonitoringSystem>();
    }

    // To trigger the Exercise Challenge Entering while player is moving toward the trigger plane/point.

    private void Update()
    {
        //inExerciseChallenge: Trigger Collision already, exChallengeSignalReceivedByPoseModel: Pose model is prepared to capture exercise posture
        if (inExerciseChallenge && player_movement.HaveChallengeSignalReceivedByPoseModel())
        {
            if (test_randowmDetermination == 0)
            {
                Debug.Log($"test push up time: {push_up_time}");
                Debug.Log($"test remaining time to do: {remaining_push_up_time}");
                push_up_time = player_movement.GetPushUpTotalTime(); // Get the real time time
                Debug.Log($" push_up_time: {push_up_time}");
                pushup_real_time_streaming.texture = player_movement.ReadTexture(); // Read newest frame from the pose model


                remaining_push_up_time -= Time.unscaledDeltaTime;

                if (push_up_time >= push_up_require) // success after doing required number of push up and keep the game going 
                {
                    inExerciseChallenge = false;
                    pushUpUI.SetActive(false);
                    ExerciseChallenge.SetActive(false);

                    fatigueMonitoringSystem.CompleteExerciseChallenge();

                    StartCoroutine(CountDown(true));
                }

                //Update UI
                pushUpUI_count.text = push_up_time.ToString();
                pushUpUI_timeleft.text = remaining_push_up_time.ToString("F1");

                if (remaining_push_up_time <= 0) // Cannot do exercise within a limited time
                {
                    inExerciseChallenge = false;
                    pushUpUI.SetActive(false);
                    ExerciseChallenge.SetActive(false);

                    fatigueMonitoringSystem.FailExerciseChallenge();

                    StartCoroutine(CountDown(false));
                }
                else if (remaining_warning_time == (float)Math.Ceiling(remaining_push_up_time)) // Giving last 3 seconds warning if player still havent finished challenge.
                {
                    remaining_warning_time--;
                    time_warning.Play();
                }
            }
            else
            {

                remaining_bridge_time -= Time.unscaledDeltaTime;
                bridge_real_time_streaming.texture = player_movement.ReadTexture();  // Read newest frame from the pose model

                if (bridge_time >= bridge_require) // success after doing required seconds of bridge and keep the game going 
                {
                    inExerciseChallenge = false;
                    bridgeUI.SetActive(false);
                    ExerciseChallenge.SetActive(false);

                    fatigueMonitoringSystem.CompleteExerciseChallenge();

                    StartCoroutine(CountDown(true));
                }


                //Update UI
                bridgeUI_count.text = remaining_bridge_time.ToString("F1");

                if (remaining_bridge_time <= 0) // Cannot do exercise within a limited time
                {
                    inExerciseChallenge = false;
                    pushUpUI.SetActive(false);
                    ExerciseChallenge.SetActive(false);

                    fatigueMonitoringSystem.FailExerciseChallenge();

                    StartCoroutine(CountDown(false));
                }
                else if (remaining_warning_time == (float)Math.Ceiling(remaining_bridge_time)) // Giving last 3 seconds warning if player still havent finished challenge.
                {
                    remaining_warning_time--;
                    time_warning.Play();
                }
            }
        }
    }

    void OnTriggerEnter(Collider other)
    {
        if (other.gameObject.tag == "Player" && !isTriggered)
        {
            isTriggered = true;

            player_movement.ChangeToExerciseChallenge();
            pausor.PauseInGameState();
            ingameRealTimeStreaming.SetActive(false); 
            ExerciseChallenge.SetActive(true);

            // If the challenge is push up
            if (test_randowmDetermination == 0)
            {
                pushUpUI.SetActive(true);

                inExerciseChallenge = true;
            }
            else // test == 1
            {
                bridgeUI.SetActive(true);

                inExerciseChallenge = true;
            }
        }
    }


    //Restarting screen/animation when ingame is restarted from exercise challenge mode
    IEnumerator CountDown(bool isSucceeded)
    {
        gameRestartCountingExercise.SetActive(true);
        ingameRealTimeStreaming.SetActive(true);// Let the camera in normal running mode be active again
        push_up_time = 0;

        if (isSucceeded) {
            success_image.SetActive(true);

            exercise_succeed.ignoreListenerPause = true;
            exercise_succeed.Play();
            yield return new WaitForSecondsRealtime(3f);

            success_image.SetActive(false);
        }
        else
        {
            fail_image.SetActive(true);

            exercise_fail.ignoreListenerPause = true;
            exercise_fail.Play();
            yield return new WaitForSecondsRealtime(3f);

            fail_image.SetActive(false);
        }
        yield return new WaitForSecondsRealtime(0.2f);
        count_down_3.SetActive(true);
        count_down_sound_effect.Play();

        yield return new WaitForSecondsRealtime(1);
        count_down_2.SetActive(true);
        count_down_sound_effect.Play();

        yield return new WaitForSecondsRealtime(1);
        count_down_1.SetActive(true);
        count_down_sound_effect.Play();

        yield return new WaitForSecondsRealtime(1);
        count_down_exercise.SetActive(true);
        game_start_sound_effect.Play();

        count_down_3.SetActive(false);
        count_down_2.SetActive(false);
        count_down_1.SetActive(false);

        yield return new WaitForSecondsRealtime(1);
        count_down_exercise.SetActive(false);

        gameRestartCountingExercise.SetActive(false);

        player_movement.ChangeFromExerciseChallenge();
        pausor.ResumeInGameState();
    }
}
