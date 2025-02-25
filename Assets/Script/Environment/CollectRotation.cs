using UnityEngine;

public class CollectRotation : MonoBehaviour
{

    [SerializeField] private int rotateSpeed = 60;
    // Update is called once per frame
    void Update()
    {
        transform.Rotate(0, rotateSpeed * Time.deltaTime, 0, Space.World);
    }
}
