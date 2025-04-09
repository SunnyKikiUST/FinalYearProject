using TMPro;
using UnityEngine;
using UnityEngine.UI;
using System.Collections;

public class FatigueMonitoringSystem : MonoBehaviour
{
    [Header("Exhaustion Setting")]
    [Range(0, 100)]
    [SerializeField] private float exhaustion_score = 0f;
    private float previous_exhaustion_score = 0f;
    [SerializeField] private float exhaustion_increase_rate = 10f;
    [SerializeField] private float exhaustion_increase_rate_exercise = 15f;
    [SerializeField] private float exhaustion_decrease_rate = 3f;
    [SerializeField] private float exhaustion_decrease_rate_exercise = 5f;
    [SerializeField] private float increase_diff_time_threahold = 3f; // time threshoold for increase time

    [Header("Performance Tracking")]
    [SerializeField] private int consecutive_obstacles_passed = 0;

    [Header("UI Element")]
    [SerializeField] private TextMeshProUGUI exhaustion_score_text;
    [SerializeField] private GameObject exhaustion_increment_text;
    [SerializeField] private GameObject exhaustion_decrement_text;
    [SerializeField] private Slider exhaustion_score_slider;

    private float difficulty_increase_timer = 0f;
    private bool shouldIncreaseDifficulty = false;
    private float percentage_gain_for_speed = 0;
    private GameObject player;
    private ObstacleSpawner obstacle_spawner;

    private int previous_exhaustion_level = -1;

    public static FatigueMonitoringSystem Instance = null; 

    void Awake()
    {
        // Initialize the singleton instance. If one exists, ensure only one exists.
        if (Instance == null)
        {
            Instance = this;
        }
        else
        {
            Destroy(gameObject);
        }
    }

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        player = GameObject.Find("Player");
        obstacle_spawner = GameObject.Find("LevelControl").GetComponent<ObstacleSpawner>();

        previous_exhaustion_score = exhaustion_score;
    }

    // Update is called once per frame
    void Update()
    {
        // GameOver due to high exhaustion level
        if(exhaustion_score >= 100)
        {
            CanGameOver();
        }
        else
        {
            CannotGameOver();
        }

        // Increase difficulty timer when in fresh state
        if(exhaustion_score >= 0 && exhaustion_score <= 20)
        {
            difficulty_increase_timer += Time.deltaTime;
            if(difficulty_increase_timer >= increase_diff_time_threahold) // After increase_diff_time_threahold second
            {
                Debug.Log($"fatigue increased speed increase difficulty");
                shouldIncreaseDifficulty = true;
                difficulty_increase_timer = 0f;
            }
        }

        UpdateDifficulty();
    }

    public void PassThroughObstacle()
    {
        consecutive_obstacles_passed += 1;
        Debug.Log($"fatigue increased speed consecutive_obstacles_passed:{consecutive_obstacles_passed}");
        if (consecutive_obstacles_passed >= 5)
        {
            StartCoroutine(ShowExhaustionScoreChange(false));
            exhaustion_score -= exhaustion_decrease_rate;
            consecutive_obstacles_passed %= 5; // To prevent the UpdateDifficulty does not reflect instantly 
        }
        
    }

    // Call this when player fails to pass through an obstacle
    public void FailObstacle()
    {
        exhaustion_score += exhaustion_increase_rate;
        consecutive_obstacles_passed = 0;
        UpdateDifficulty();
    }

    // Not yet used
    public void CompleteExerciseChallenge()
    {
        // Player comlpeted exericse quickly -> less exhaustion 
        exhaustion_score -= exhaustion_decrease_rate_exercise;
    }

    // Not yet used
    public void FailExerciseChallenge()
    {
            exhaustion_score += exhaustion_increase_rate_exercise;
            UpdateDifficulty();
    }

    // Helper to determine previous exhaustion category
    private int GetExhasutionLevel()
    {
        if (exhaustion_score <= 20) return 0; // Fresh
        if (exhaustion_score <= 40) return 1; // Slightly Tired
        if (exhaustion_score <= 60) return 2; // Fatigue
        if (exhaustion_score <= 80) return 3; // Very Tired
        return 4; // Exhausted
    }

    private bool HasChangedExhaustionLevel()
    {
        int current_level = GetExhasutionLevel();

        if (previous_exhaustion_level == -1)
        {
            previous_exhaustion_level = current_level;
            return true; // For time execute
        }

        if(current_level != previous_exhaustion_level)
        {
            Debug.Log("Change exhaustion level");
            int old_level = previous_exhaustion_level;
            previous_exhaustion_level = current_level;

            return true;
        }

        return false;
    }

    private void UpdateDifficulty()
    {
        exhaustion_score = Mathf.Clamp(exhaustion_score, 0f, 100f);

        // Apply difficulty adjustments based on exhaustion level
        if (GetExhasutionLevel() == 0)
        {
            // Initial setting OR Reset configuration while going back to the exhaustion level 0
            if (HasChangedExhaustionLevel())
            {
                Debug.Log("wasAboveFreshState 5"); //have bug

                player.GetComponent<PlayerMovementWithMVEstimationTest>().SetBackToDefaultSpeed();
                //player.GetComponent<PlayerMovement>().SetBackToDefaultSpeed();
                obstacle_spawner.SetBackToDefaultInterval();
                percentage_gain_for_speed = 0;
                obstacle_spawner.SetBackToDefaultInterval();
            }

            // Increase difficulty 
            if (shouldIncreaseDifficulty)
            {
                Debug.Log("fatigue player_speed increasing difficulty");
                percentage_gain_for_speed += 5;
                percentage_gain_for_speed = player.GetComponent<PlayerMovementWithMVEstimationTest>().IncreaseSpeedFromDefault(percentage_gain_for_speed);
                //percentage_gain_for_speed = player.GetComponent<PlayerMovement>().IncreaseSpeedFromDefault(percentage_gain_for_speed);
                shouldIncreaseDifficulty = false;
            }
        }
        else if(GetExhasutionLevel() == 1)
        {
            if (HasChangedExhaustionLevel())
            {
                // Slightly Tired
                player.GetComponent<PlayerMovementWithMVEstimationTest>().DecreaseSpeedFromDefault(5); // 5% of default speed decrease from default speed
                //player.GetComponent<PlayerMovement>().DecreaseSpeedFromDefault(5); // 5% of default speed decrease from default speed
                obstacle_spawner.IncreaseSpawnIntervalFromDefault(0.1f);
            }

            // Now, we don't increase difficulty(i.e. increase speed) since the player is assumed to be slightly tired from fresh.
            if (previous_exhaustion_score <= 20)
            {
                Debug.Log("wasAboveFreshState 4");
                shouldIncreaseDifficulty = false;
                difficulty_increase_timer = 0f;
            }
        }
        else if (GetExhasutionLevel() == 2)
        {
            if (HasChangedExhaustionLevel())
            {
                // Fatigue
                player.GetComponent<PlayerMovementWithMVEstimationTest>().DecreaseSpeedFromDefault(10);
                //player.GetComponent<PlayerMovement>().DecreaseSpeedFromDefault(10); // 10% of default speed decrease from default speed
                obstacle_spawner.IncreaseSpawnIntervalFromDefault(0.2f);
            }
        }
        else if (GetExhasutionLevel() == 3)
        {
            if (HasChangedExhaustionLevel())
            {
                // Very Tired
                player.GetComponent<PlayerMovementWithMVEstimationTest>().DecreaseSpeedFromDefault(15);
                //player.GetComponent<PlayerMovement>().DecreaseSpeedFromDefault(15); // 15% of default speed decrease from default speed
                obstacle_spawner.IncreaseSpawnIntervalFromDefault(0.4f);
            }
        }
        else if (GetExhasutionLevel() == 4)
        {
            if (HasChangedExhaustionLevel())
            {
                // Exhausted
                player.GetComponent<PlayerMovementWithMVEstimationTest>().DecreaseSpeedFromDefault(20);
                //player.GetComponent<PlayerMovement>().DecreaseSpeedFromDefault(20); // 20% of default speed decrease from default speed
                obstacle_spawner.IncreaseSpawnIntervalFromDefault(0.5f);
            }
        }

        // Update any game elements that need to reflect these changes
        UpdateGameUI();

        // For FailObstacle method
        previous_exhaustion_score = exhaustion_score;
    }


    // Update game objects based on difficulty changes
    private void UpdateGameUI()
    {
        exhaustion_score_text.text = $"{exhaustion_score}";
        SetExhaustionScore(exhaustion_score);
    }

    // Hitting obstacle and fail to do exercise will be gameover after call this
    private void CanGameOver()
    {
        ObstacleCollision.CanGameOver();
        // Exercise part ...
    }

    private void CannotGameOver()
    {
        ObstacleCollision.CannotGameOver();
        // Exercise part ...
    }

    public void UseShowExhastionScoreChangeWithCoroutine(bool isIncremental)
    {
        StartCoroutine(ShowExhaustionScoreChange(isIncremental));
    }

    // Set the slider based on exhaustion score
    private void SetExhaustionScore(float exhaustion_score)
    {
        exhaustion_score_slider.value = exhaustion_score;
    }

    IEnumerator ShowExhaustionScoreChange(bool isIncremental)
    {

        if (isIncremental)
        {
            exhaustion_increment_text.GetComponent<TextMeshProUGUI>().text = $"+ {exhaustion_increase_rate} fatigue !";
            exhaustion_increment_text.SetActive(false);
            exhaustion_increment_text.SetActive(true);
            //yield return new WaitForSecondsRealtime(2);
            yield break;
        }
        else
        {
            exhaustion_decrement_text.GetComponent<TextMeshProUGUI>().text = $"- {exhaustion_decrease_rate} fatigue !";
            exhaustion_decrement_text.SetActive(false);
            exhaustion_decrement_text.SetActive(true);
            //yield return new WaitForSecondsRealtime(2);
            yield break;
        }
    }
}
