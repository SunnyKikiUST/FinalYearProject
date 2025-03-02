using UnityEngine;

public class InGameAudioController : MonoBehaviour
{

    [SerializeField] private AudioSource ingame_bgm;
    [SerializeField] private AudioSource gameover_fx;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        // For coin sound setup, it is implemented in CollectSound script.
        ingame_bgm.volume = PlayerPrefs.GetFloat("InGame_BGM_Volume", 0.2f);
    }

    public void PlayGameOverFX()
    {
        gameover_fx.Play();
    }

    public void StopInGameBGM()
    {
        ingame_bgm.Stop();
    }

}
