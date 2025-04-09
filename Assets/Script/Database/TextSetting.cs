using SharpCompress.Common;
using System.IO;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class TextSetting : MonoBehaviour
{
    [SerializeField] private TMP_InputField open_AIKey_InputField;
    [SerializeField] private TMP_InputField webcam_Index_InputField;
    private string open_AIKey = "";
    private string webcam_Index = "";


    // Should be store in C:/Users/hero/AppData/LocalLow/DefaultCompany/My project. If not, then it is based on Application.persistentDataPath
    private string file_path;

    //private void Start()
    //{
    //    Debug.Log(Application.persistentDataPath);
    //}

    // Store the string setting and use in python script
    void Awake()
    {
        file_path = Path.Combine(Application.persistentDataPath, "settings.txt");

        if (File.Exists(file_path))
        {
            string[] settings = File.ReadAllLines(file_path);
            if (settings.Length >= 2)
            {
                // Get the original settings. If player enter empty string in input field(s),
                // the setting(s) will be used again while saving to settings.txt
                open_AIKey = settings[0];
                webcam_Index = settings[1];
            }
        }
        else
        {
            // array of lines to write
            string[] lines = new string[] { open_AIKey, webcam_Index };

            // Write lines to the text file
            File.WriteAllLines(file_path, lines);

            Debug.Log("Settings saved to: " + file_path);
        }
    }

    private void Start()
    {
        open_AIKey_InputField.text = PlayerPrefs.GetString("OpenAI_api_key", "");
        webcam_Index_InputField.text = PlayerPrefs.GetString("Webcam_index", "");
    }

    public string GetOpenAIKey()
    {
        return open_AIKey;
    }

    public string GetWebcamIndex()
    {
        return webcam_Index;
    }


    // Call this method via "back" button in OptionMenu to save the settings.
    public void SaveSettings()
    {
        string previous_open_AIKey = open_AIKey;
        string previous_webcam_Index = webcam_Index;

        if (open_AIKey_InputField.text != open_AIKey)
        {
            PlayerPrefs.SetString("OpenAI_api_key", open_AIKey_InputField.text);
            open_AIKey = open_AIKey_InputField.text;
        }


        if (webcam_Index_InputField.text != webcam_Index)
        {
            PlayerPrefs.SetString("Webcam_index", webcam_Index_InputField.text);
            webcam_Index = webcam_Index_InputField.text;
        }

        if(previous_open_AIKey != open_AIKey || previous_webcam_Index != webcam_Index)
        {
            // array of lines to write
            string[] lines = new string[] { open_AIKey, webcam_Index };

            // Write lines to the text file
            File.WriteAllLines(file_path, lines);

            Debug.Log("Settings saved to: " + file_path);
        }
    }
}
