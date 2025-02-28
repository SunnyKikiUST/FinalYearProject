using UnityEngine;
using UnityEngine.UI;


// This class store the local preference setting that are adjusted by player in Option Page.
public class PlayerPreference : MonoBehaviour
{

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        float volume = PlayerPrefs.GetFloat("Menu_BGM_Volume", -1);
        AudioSource bgm = GameObject.Find("BGM").GetComponent<AudioSource>();

        // Check is Menu_BGM_Volume saved in player preference.
        // Since the this.gameobject is in Menu scene. The relevant AudioSource is also assigned with a volume.
        if (volume == -1)
        {
            PlayerPrefs.SetFloat("Menu_BGM_Volume", 0.2f);
            bgm.volume = 0.2f;
        }
        else
        {
            bgm.volume = volume;
        }

        // Check is InGame_BGM_Volume saved in player preference.
        volume = PlayerPrefs.GetFloat("InGame_BGM_Volume", -1);
        if (volume == -1)
        {
            PlayerPrefs.SetFloat("InGame_BGM_Volume", 0.2f);
        }

        // Check is Coin_Collect_Sound saved in player preference.
        volume = PlayerPrefs.GetFloat("Coin_Collect_Sound");
        if (volume == -1)
        {
            PlayerPrefs.SetFloat("Coin_Collect_Sound", 0.2f);
        }

    }


    // This method is referenced at the Slider manually in OptionUI gameobject of Menu scene (On Value Changed)
    public void UpdateMenuBGMVolume(Slider bgm_vol_slider)
    {      
        PlayerPrefs.SetFloat("Menu_BGM_Volume", bgm_vol_slider.value);

        AudioSource bgm = GameObject.Find("BGM").GetComponent<AudioSource>();
        bgm.volume = bgm_vol_slider.value;
    }

    // This method is referenced at the Slider manually in InGame Scene (On Value Changed)
    // Since the setting is adjusted in Menu scene. Unlike UpdateMenuBGMVolume method, the actual ingame bgm will be changed in Game scene.
    public void UpdateInGameBGMVolume(Slider bgm_vol_slider)
    {
       PlayerPrefs.SetFloat("InGame_BGM_Volume", bgm_vol_slider.value);
    }


    // This method is referenced at the Slider manually in InGame Scene (On Value Changed)
    // Since the setting is adjusted in Menu scene. Unlike UpdateMenuBGMVolume method, the actual ingame bgm will be changed in Game scene.
    public void UpdateCoinCollectSound(Slider coin_sound_slider)
    {
        PlayerPrefs.SetFloat("Coin_Collect_Sound", coin_sound_slider.value);
    }
}
