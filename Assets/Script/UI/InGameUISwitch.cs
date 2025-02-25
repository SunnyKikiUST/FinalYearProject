using UnityEngine;
using UnityEngine.UI;
using TMPro;
using UnityEngine.SceneManagement;
using System.Drawing;

public class InGameUISwitch : MonoBehaviour
{

    [SerializeField] private GameObject inGameUI;
    [SerializeField] private GameObject resultScreen;

    [SerializeField] private TextMeshProUGUI match_coin_number;
    [SerializeField] private TextMeshProUGUI match_score_number;
    [SerializeField] private TextMeshProUGUI point;
    [SerializeField] private TextMeshProUGUI break_record_text;
    [SerializeField] private Button replay_button;
    [SerializeField] private Button menu_button;

    public static InGameUISwitch Instance;
    private void Start()
    {
        replay_button.onClick.AddListener(OnReplayClicked);
        menu_button.onClick.AddListener(OnBackToManuClicked);
    }
    private void Awake()
    {
        // Initialize the singleton instance. If one exists, ensure only one exists.
        if (Instance == null)
        {
            Instance = this;
        }
    }

    // Switch to Result Screen if game loses. This method is using in ObstacleCollision script.
    public void SwitchToResultScreen()
    {
        inGameUI.SetActive(false);

        match_coin_number.text = CollectableControl.GetCurrentCoin().ToString();
        match_score_number.text = LevelScore.GetCurrentScore().ToString();

        //Debug.Log($"CollectableControl.GetCurrentCoin().ToString(): {CollectableControl.GetCurrentCoin().ToString()}");
        //Debug.Log($"LevelScore.GetCurrentScore().ToString() {LevelScore.GetCurrentScore().ToString()}");

        int match_point = CollectableControl.GetCurrentCoin() + LevelScore.GetCurrentScore();
        point.text = match_point.ToString();

        CollectableControl.CoinToZero();
        LevelScore.ScoreToZero();

        if (CurrentCoinAndScore.GetBestPoint() > match_point)
        {
            break_record_text.text = "";
        }

        // TODO: Implement update new record to database.

        resultScreen.SetActive(true);
    }

    // Play again
    private void OnReplayClicked()
    {
        SceneManager.LoadScene(SceneManager.GetActiveScene().buildIndex);
    }

    // Go back to main menu
    // TODO: To see how to start from main menu page directly while switching scene
    private void OnBackToManuClicked()
    {
        SceneManager.LoadScene("Menu");
    }
}
