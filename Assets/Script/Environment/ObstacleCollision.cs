using UnityEngine;
using System.Collections;

public class ObstacleCollision : MonoBehaviour
{
    static private bool playerHasCollided = false;
    static private bool canGameOver = false;
    private bool obstacleHasCollided = false;
    private bool obstacleHasPassThroughPlayer = false;

    private ParticleSystem smoke_VFX;

    [SerializeField] private GameObject player;

    private InGameAudioController controller;
    private void Start()
    {
        player = GameObject.Find("Player");

        smoke_VFX = player.transform.Find("smoke_VFX")?.gameObject.GetComponent<ParticleSystem>();

        controller = GameObject.Find("LevelControl/Audio/BGM").GetComponent<InGameAudioController>();
    }

    // The script is attached to all obstacles.
    void OnTriggerEnter(Collider other)
    {
        if(other.gameObject.tag == "Player" && !playerHasCollided && canGameOver)
        {
            // To prevent other obstacle to collide with player again.
            playerHasCollided = true;
            // To prevent the action of PassThroughObstacle() in Update()
            obstacleHasCollided = true;

            // Stop character moving
            // other.gameObject.GetComponent<PlayerMovementWithMVEstimation>().enabled = false;
            other.gameObject.GetComponent<PlayerMovement>().StopOnCollision();
            other.gameObject.GetComponent<PlayerMovement>().enabled = false;


            Debug.Log($"collision happened: {other}");
            Debug.Log($"collision happened start from: {this}");

            // Perform animation 'Stumble Backwards'
            Transform child_transform = other.gameObject.transform.Find("Donna");
            if (child_transform != null) child_transform.gameObject.GetComponent<Animator>().Play("Stumble Backwards");

            // Stop the dynamic obstacle if the collision is triggered with it
            if (this.gameObject.GetComponent<MovingObstacleMotion>() != null)
                this.gameObject.GetComponent<MovingObstacleMotion>().enabled = false;


            // Stop obstacle Spawner and Coin Spawner
            ObstacleSpawner obstacle_spawner = GameObject.Find("LevelControl").GetComponent<ObstacleSpawner>();
            obstacle_spawner.StopCoroutine();
            // Stop all dynamic obstacles
            foreach (GameObject dynamic_obstacle in obstacle_spawner.GetDynamicObstacles())
            {
                if(dynamic_obstacle == null) continue;
                dynamic_obstacle.GetComponent<MovingObstacleMotion>()?.StopMoving();
            }
            obstacle_spawner.enabled = false;
            GameObject level_control = GameObject.Find("LevelControl");
            level_control.GetComponent<CoinSpawner>().StopCoroutine();
            level_control.GetComponent<CoinSpawner>().enabled = false;
            level_control.GetComponent<LevelScore>().enabled = false;
            InGameAudioController controller = GameObject.Find("LevelControl/Audio/BGM").GetComponent<InGameAudioController>();
            controller.StopInGameBGM();
            controller.PlayGameOverFX();

            StartCoroutine(DelayedSwitchToResultScreen());
        }
        // Increase exhaustion score in fatigue monitoring system.
        else if (!canGameOver)
        {
            if(smoke_VFX != null) smoke_VFX.Play();
            controller.PlayHurtFX();
            gameObject.SetActive(false);
            FatigueMonitoringSystem.Instance.FailObstacle();
        }
    }

    private void Update()
    {
        if(
            gameObject.transform.position.z < player.transform.position.z && 
            !obstacleHasCollided &&
            !obstacleHasPassThroughPlayer &&
            gameObject.GetComponent<ObstacleRemoval>() != null
            )
        {
            obstacleHasPassThroughPlayer = true;

            //Debug.Log($"fatigue 123 PassThroughObstacle gameObject.transform.position.z from ObstacleCollision: {gameObject.transform.position.z}");

            FatigueMonitoringSystem.Instance.PassThroughObstacle();
        }
    }

    private IEnumerator DelayedSwitchToResultScreen()
    {
        yield return new WaitForSeconds(2);

        // Switch UI to Result Screen
        InGameUISwitch.Instance.SwitchToResultScreen();
    }

    public static void ResetCollision()
    {
        playerHasCollided = false;
        canGameOver = false;
    }

    public static void CanGameOver()
    {
        canGameOver = true;
    }

    public static void CannotGameOver()
    {
        canGameOver = false;
    }
}
