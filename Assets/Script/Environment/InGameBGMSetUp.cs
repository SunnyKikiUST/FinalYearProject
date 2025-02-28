using UnityEngine;

public class InGameBGMSetUp : MonoBehaviour
{

    [SerializeField] private AudioSource ingame_bgm;
    [SerializeField] private AudioSource coin_sound;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        // For coin sound setup, it is implemented in CollectSound script.
        ingame_bgm.volume = PlayerPrefs.GetFloat("InGame_BGM_Volume", 0.2f);
    }
}
