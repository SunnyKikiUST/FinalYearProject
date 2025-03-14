using System.Collections;
using UnityEngine;

public class ExerciseFinishingRestarter : MonoBehaviour
{
    [SerializeField] private GameObject count_down_3;
    [SerializeField] private GameObject count_down_2;
    [SerializeField] private GameObject count_down_1;
    [SerializeField] private GameObject count_down_exercise;
    [SerializeField] private AudioSource count_down_sound_effect;
    [SerializeField] private AudioSource game_start_sound_effect;

    void Start()
    {
        StartCoroutine(CountDown());
    }

    //Restarting screen/animation when ingame is restarted from exercise challenge mode
    IEnumerator CountDown()
    {
        yield return new WaitForSecondsRealtime(0.2f);
        count_down_3.SetActive(true);
        count_down_sound_effect.Play();

        yield return new WaitForSecondsRealtime(1);
        count_down_2.SetActive(true);
        count_down_sound_effect.Play();

        yield return new WaitForSecondsRealtime(1);
        count_down_1.SetActive(true);
        count_down_sound_effect.Play();

        yield return new WaitForSecondsRealtime(1);
        count_down_exercise.SetActive(true);
        game_start_sound_effect.Play();

        count_down_3.SetActive(false);
        count_down_2.SetActive(false);
        count_down_1.SetActive(false);

        yield return new WaitForSeconds(1);
        count_down_exercise.SetActive(false);
    }
}
