using UnityEngine;

public class MovingObstacleMotion : MonoBehaviour
{
    private float speed = 10f;

    // Update is called once per frame
    void Update()
    {
        transform.position += Vector3.back * speed * Time.deltaTime;
    }

    public void StopMoving()
    {
        speed = 0;
    }

    public void StartMoving()
    {
        speed = 10f;
    }
}
