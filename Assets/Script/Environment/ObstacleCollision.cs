using UnityEngine;
using System.Collections;

public class ObstacleCollision : MonoBehaviour
{


    void OnTriggerEnter(Collider other)
    {
        if(other.gameObject.tag == "Player")
        {
            //Debug.Log($"collision happened: {other}");
            //Debug.Log($"collision happened start from: {this}");

            //player.GetComponent<PlayerMovementWithMVEstimation>().enabled = false;

            // Stop character moving
            other.gameObject.GetComponent<PlayerMovement>().enabled = false;

            // Perform animation 'Stumble Backwards'
            Transform child_transform = other.gameObject.transform.Find("Donna");
            if (child_transform != null) child_transform.gameObject.GetComponent<Animator>().Play("Stumble Backwards");

            // Stop the dynamic obstacle if the collision is triggered with it
            if (this.gameObject.GetComponent<MovingObstacleMotion>() != null)
                this.gameObject.GetComponent<MovingObstacleMotion>().enabled = false;

            // Stop obstacle Spawner and Coin Spawner
            GameObject.Find("LevelControl").GetComponent<ObstacleSpawner>().StopCoroutine();
            GameObject.Find("LevelControl").GetComponent<ObstacleSpawner>().enabled = false;
            GameObject.Find("LevelControl").GetComponent<CoinSpawner>().StopCoroutine();
            GameObject.Find("LevelControl").GetComponent<CoinSpawner>().enabled = false;
            GameObject.Find("LevelControl").GetComponent<LevelScore>().enabled = false;

            StartCoroutine(DelayedSwitchToResultScreen());
        }
    }

    private IEnumerator DelayedSwitchToResultScreen()
    {
        yield return new WaitForSeconds(2);

        // Switch UI to Result Screen
        InGameUISwitch.Instance.SwitchToResultScreen();
    }
}
