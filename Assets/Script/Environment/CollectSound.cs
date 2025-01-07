using UnityEngine;

public class CollectSound : MonoBehaviour
{
    [SerializeField] AudioSource sound;

    // Update is called once per frame
    void OnTriggerEnter(Collider other)
    {
        sound.Play();

        //gameObject.SetActive(false);
        Destroy(gameObject);
    }
}
