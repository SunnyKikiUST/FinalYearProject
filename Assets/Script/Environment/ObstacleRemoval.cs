using UnityEngine;

public class ObstacleRemoval : MonoBehaviour
{
    private void OnBecameInvisible()
    {
        Debug.Log($"Obstacle {gameObject} has removed.");
        Destroy(gameObject);
    }
}
