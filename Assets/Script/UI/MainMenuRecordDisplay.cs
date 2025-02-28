using UnityEngine;
using TMPro;

public class MainMenuRecordDisplay : MonoBehaviour
{
    [SerializeField] private TextMeshProUGUI highest_coin_text;
    [SerializeField] private TextMeshProUGUI highest_score_text;
    [SerializeField] private TextMeshProUGUI best_point_text;


    // Display record in main menu.
    public void DisplayRecordInMenu()
    {
        highest_coin_text.text = CurrentCoinAndScore.GetHighestCoin().ToString();
        highest_score_text.text = CurrentCoinAndScore.GetHighestScore().ToString();
        best_point_text.text = CurrentCoinAndScore.GetBestPoint().ToString();
    }

    void Start()
    {
        DisplayRecordInMenu();
    }
}
