using TMPro;
using UnityEngine;
using UnityEngine.UI;
using System.Collections;
using UnityEngine.SocialPlatforms.Impl;

public class LevelScore : MonoBehaviour
{
    [SerializeField] private GameObject score_display;
    [SerializeField] private float duration = 1f;
    private static int score = 0;
    private float timer = 0f;
    private static bool start_counting = false;

    void Update()
    {
        if(start_counting) timer += Time.deltaTime;

        // Increase score for each full interval elapsed.
        if (timer >= duration)
        {
            // Calculate how many increments to apply if more than one interval has passed.
            int increments = (int)(timer / duration);
            score += increments;
            timer = timer % duration;

            score_display.GetComponent<TextMeshProUGUI>().text = score.ToString();
        }
    }
    
    public static void StartCountingScore()
    {
        start_counting = true;
    }

    public static int GetCurrentScore()
    {
        return score;
    }

    public static void ScoreToZeroAndStop()
    {
        score = 0;
        start_counting = false;
    }
}
