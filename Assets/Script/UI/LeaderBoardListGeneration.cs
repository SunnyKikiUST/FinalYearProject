using UnityEngine;
using System.Collections.Generic;
using TMPro;

public class LeaderBoardListGeneration : MonoBehaviour
{

    [SerializeField] private GameObject scroll_view_content;
    [SerializeField] private GameObject template_object;
    [SerializeField] private TextMeshProUGUI player_ranking; 

    // Load the leaderboard list when user enter the leaderboard page.
    public async void LoadLeaderBoard()
    {
        List<DatabaseManager.Record> list = await DatabaseManager.Instance.GetTop50UserRecordAsync();
        for (int i = 0; i < list.Count; i++)
        {

            // Setting new object be child of the leaderboard list
            GameObject entry = Instantiate(template_object, scroll_view_content.transform);
            string[] array = new string[entry.transform.childCount];
            int index = 0; // Index to store child names in the array
            foreach (Transform child in entry.transform)
            {
                array[index] = child.name;
                index++;
            }

            // Get the text components using child names:
            TMP_Text position_text = entry.transform.Find(array[0])?.GetChild(0).gameObject.GetComponent<TMP_Text>();
            TMP_Text name_text = entry.transform.Find(array[1])?.GetChild(0).gameObject.GetComponent<TMP_Text>();
            TMP_Text coin_text = entry.transform.Find(array[2])?.GetChild(0).gameObject.GetComponent<TMP_Text>();
            TMP_Text score_text = entry.transform.Find(array[3])?.GetChild(0).gameObject.GetComponent<TMP_Text>();

            if (position_text != null) position_text.text = (i + 1).ToString();
            if (name_text != null) name_text.text = list[i].user_name;
            if (coin_text != null) coin_text.text = list[i].coins.ToString();
            if (score_text != null) score_text.text = list[i].score.ToString();

            if (list[i].user_name == CurrentCoinAndScore.GetUserName())
            {
                player_ranking.text = (i + 1).ToString();
            }
        }
    }

    // Remove the leaderboard list in UI to make the player get the latest list each time.
    public void RemoveLeaderBoard()
    {
        foreach (Transform child in scroll_view_content.transform)
        {
            GameObject.Destroy(child.gameObject);
        }
    }

    public void TestFunction()
    {
        int a = 5;
    }
}
