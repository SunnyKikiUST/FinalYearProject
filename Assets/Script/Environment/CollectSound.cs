using UnityEngine;
using UnityEngine.UI;

public class CollectSound : MonoBehaviour
{
    [SerializeField] AudioSource sound;

    // Update is called once per frame
    void OnTriggerEnter(Collider other)
    {
        if(other.tag == "Player")
        {
            CollectableControl.IncreaseCoinByOne();
            sound.Play();
        }

        //gameObject.SetActive(false);
        Destroy(gameObject);
    }
}
