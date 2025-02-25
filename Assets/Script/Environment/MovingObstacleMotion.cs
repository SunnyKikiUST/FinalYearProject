using UnityEngine;

public class MovingObstacleMotion : MonoBehaviour
{
    public float speed = 10f;


    // Update is called once per frame
    void Update()
    {
        transform.position += Vector3.back * speed * Time.deltaTime;
    }
}
