using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using TMPro;
public class LoginAccess : MonoBehaviour
{
    [SerializeField] private TextMeshProUGUI login_status_text ;
    [SerializeField] private TextMeshProUGUI register_status_text;

    [SerializeField] private Button login_button;
    [SerializeField] private Button register_button;
    [SerializeField] private Button go_login_button;
    [SerializeField] private Button go_register_button;

    [SerializeField] private TMP_InputField usernameInputField;
    [SerializeField] private TMP_InputField passwordInputField;
    [SerializeField] private TMP_InputField usernameRegInputField;
    [SerializeField] private TMP_InputField passwordRegInputField;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start(){
        // Hook up the button click event
        if (login_button != null)
        {
            login_button.onClick.AddListener(OnLoginButtonClickedAsync);
            register_button.onClick.AddListener(OnRegisterButtonClicked);
            go_login_button.onClick.AddListener(OnCleanIputField);
        }
    }

    private void OnCleanIputField()
    {
        usernameInputField.text = "";
        passwordInputField.text = "";
        usernameRegInputField.text = "";
        passwordRegInputField.text = "";
        login_status_text.text = "";
        register_status_text.text = "";
    }

    // Event: login operation 
    private async void OnLoginButtonClickedAsync()
    {
        login_status_text.text = "Processing...";
        go_register_button.enabled = false;

        string username = usernameInputField.text;
        string password = passwordInputField.text;

        // Basic validation to ensure both fields are filled
        if (string.IsNullOrEmpty(username) || string.IsNullOrEmpty(password))
        {
            login_status_text.text = "Please enter both username and password.";
            return;
        }

        // Call the authentication method from the DatabaseManager singleton
        string authenticatedResponse = await DatabaseManager.Instance.AuthenticateUser(username, password);

        
        if (authenticatedResponse == "success") //Login successfully
        {
            CurrentCoinAndScore.RetrieveRecordFromDataBase(username);
            login_status_text.text = "Login successful!";
            SceneManager.LoadScene("Menu");
        }
        else
        {
            login_status_text.text = authenticatedResponse;
        }

        go_register_button.enabled = true;
    }

    // Event: register operation 
    private async void OnRegisterButtonClicked() 
    {
        register_status_text.text = "Processing...";
        go_login_button.enabled = false;

        string username = usernameRegInputField.text.Trim();
        string password = passwordRegInputField.text;

        // Basic validation to ensure both fields are filled
        if (string.IsNullOrEmpty(username) || string.IsNullOrEmpty(password))
        {
            register_status_text.text = "Please enter both username and password.";
            return;
        }

        // Call the registration method from the DatabaseManager singleton
        string registeredResponse = await DatabaseManager.Instance.RegisterUserAsync(username, password);

        if (registeredResponse == "Register successful!") //Register successfully
        {
            register_status_text.text = registeredResponse;
        }
        else
        {
            register_status_text.text = registeredResponse;
        }
        go_login_button.enabled = true;
    }
}
