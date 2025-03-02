using UnityEngine;
using System.Collections;

public class ObstacleCollision : MonoBehaviour
{
    static private bool hasCollided = false; 
    // The script is attached to all obstacles.
    void OnTriggerEnter(Collider other)
    {
        if(other.gameObject.tag == "Player" && !hasCollided)
        {
            hasCollided = true;
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
            GameObject.Find("LevelControl").GetComponent<CoinSpawner>().StopCoroutine();
            GameObject.Find("LevelControl").GetComponent<CoinSpawner>().enabled = false;
            GameObject.Find("LevelControl").GetComponent<LevelScore>().enabled = false;
            InGameAudioController controller = GameObject.Find("LevelControl/Audio/BGM").GetComponent<InGameAudioController>();
            controller.StopInGameBGM();
            controller.PlayGameOverFX();

            StartCoroutine(DelayedSwitchToResultScreen());
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
        hasCollided = false;
    }
}
