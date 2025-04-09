using SharpCompress.Common;
using System.IO;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class OptionSliderUpdate : MonoBehaviour
{
    [SerializeField] private Slider menu_BGM_slider;
    [SerializeField] private Slider ingame_BGM_slider;
    [SerializeField] private Slider coin_sound_slider;

    void Start()
    {
        menu_BGM_slider.value = PlayerPrefs.GetFloat("Menu_BGM_Volume", 0.2f);
        ingame_BGM_slider.value = PlayerPrefs.GetFloat("InGame_BGM_Volume", 0.2f);
        coin_sound_slider.value = PlayerPrefs.GetFloat("Coin_Collect_Sound", 0.2f);
    }

}
