using UnityEngine;

public class ObstacleCollision : MonoBehaviour
{
    public GameObject player;
    public GameObject character_model; //Hierarchy: "./Player/Donna"

    void OnTriggerEnter(Collider other)
    {
        Debug.Log("Collision happen");
        this.gameObject.GetComponent<BoxCollider>().enabled = false;
        player.GetComponent<PlayerMovementWithMVEstimation>().enabled = false;
        character_model.GetComponent<Animator>().Play("Stumble Backwards");
    }
}
