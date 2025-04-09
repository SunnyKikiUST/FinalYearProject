using UnityEngine;
using UnityEngine.SceneManagement;

public class MainMenu : MonoBehaviour
{
    [SerializeField] private TextSetting text_setting;
    [SerializeField] private GameObject warning;
    public void PlayGame()
    {
        if (text_setting.GetOpenAIKey() == "" || text_setting.GetWebcamIndex() == "")
        {
            warning.SetActive(true);
        }
        else
        {
            //SceneManager.LoadScene(SceneManager.GetActiveScene().buildIndex + 1);
            SceneManager.LoadScene("Game");
        }
    }

    public void QuitWarning()
    {
        warning.SetActive(false);
    }

    public void QuitGame()
    {
        Debug.Log("Quit");
        #if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false; // Stops Play Mode
        #else
        Application.Quit();
        #endif
    }
}
