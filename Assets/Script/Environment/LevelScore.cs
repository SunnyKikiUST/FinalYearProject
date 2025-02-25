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
    void Update()
    {
        timer += Time.deltaTime;

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

    public static int GetCurrentScore()
    {
        return score;
    }

    public static void ScoreToZero()
    {
        score = 0;
    }
}
