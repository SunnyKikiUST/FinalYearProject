using UnityEngine;

public class ObstacleRemoval : MonoBehaviour
{
    private GameObject player;
    [SerializeField] private float distance_delay = 10f;
    void Start()
    {
        player = GameObject.Find("Player");
        InvokeRepeating("ObstacleDestroyer", 0f, 0.1f);
    }
    private void ObstacleDestroyer()
    {
        if (gameObject.transform.position.z + distance_delay < player.transform.position.z)
        {
            //Debug.Log($"fatigue 123 Obstacle {gameObject} has removed.");
            Destroy(gameObject);
        }
    }

}
