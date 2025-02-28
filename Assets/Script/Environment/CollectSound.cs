using UnityEngine;
using UnityEngine.UI;

public class CollectSound : MonoBehaviour
{
    [SerializeField] AudioSource sound;

    private void Start()
    {
        sound.volume = PlayerPrefs.GetFloat("Coin_Collect_Sound", 0.2f);
    }

    // Update is called once per frame
    void OnTriggerEnter(Collider other)
    {
        if(other.tag == "Player")
        {
            // Increase coin number and sound at the same time
            CollectableControl.IncreaseCoinByOne();
            sound.Play();
        }

        Destroy(gameObject);
    }
}
