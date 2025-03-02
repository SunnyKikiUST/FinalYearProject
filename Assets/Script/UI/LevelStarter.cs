using System.Collections;
using UnityEngine;

public class LevelStarter : MonoBehaviour
{
    [SerializeField] private GameObject count_down_3;
    [SerializeField] private GameObject count_down_2;
    [SerializeField] private GameObject count_down_1;
    [SerializeField] private GameObject count_down_exercise;
    [SerializeField] private GameObject fadein;
    [SerializeField] private AudioSource count_down_sound_effect;
    [SerializeField] private AudioSource game_start_sound_effect;

    void Start()
    {
        StartCoroutine(CountDown());
    }

    //Starting screen/animation when ingame is started 
    IEnumerator CountDown()
    {
        yield return new WaitForSeconds(0.2f);
        count_down_3.SetActive(true);
        count_down_sound_effect.Play();

        yield return new WaitForSeconds(1);
        count_down_2.SetActive(true);
        count_down_sound_effect.Play();

        yield return new WaitForSeconds(1);
        count_down_1.SetActive(true);
        count_down_sound_effect.Play();

        yield return new WaitForSeconds(1);
        count_down_exercise.SetActive(true);
        game_start_sound_effect.Play();

        count_down_3.SetActive(false);
        count_down_2.SetActive(false);
        count_down_1.SetActive(false);
        fadein.SetActive(false);
        yield return new WaitForSeconds(1);
        count_down_exercise.SetActive(false);
    }
}
