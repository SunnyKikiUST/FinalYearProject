using UnityEngine;
using System;
using System.Diagnostics; // Start Python process
using System.Net.Sockets;
using UnityEngine.UI;
using System.Text;
using System.IO.Enumeration; // Socket communcation
using System.Threading;
using PimDeWitte.UnityMainThreadDispatcher;
using Unity.VisualScripting.Antlr3.Runtime;
using TMPro;
using System.Collections;
using System.IO;
using UnityEngine.Networking;

public class EmotionRecognition : MonoBehaviour
{

    private PlayerMovementWithMVEstimationTest player_movement;

    private bool connectedToSocket = false;
    private bool isReceivingData = true;
    private bool isConnectedToPoseModel = false;

    private TcpClient client = null;
    private NetworkStream stream = null;
    private Process python_process = null;
    private Texture2D texture = null;
    private string ADDRESS = "127.0.0.1";
    private int PORT = 65452; // Port for emotion recngnition model
    private int READ_TIME_OUT = 50000;

    private int signal = -1;


    void Start()
    {
        player_movement = GameObject.Find("Player").GetComponent<PlayerMovementWithMVEstimationTest>();

        // Establish socket socket communication
        StartCoroutine(ConnectToSocket());

        // Use thread to receive movement and frame
        Thread thread = new Thread(GetQuoteReadySignal);
        thread.IsBackground = true;
        thread.Start();
    }

    IEnumerator ConnectToSocket()
    {
        int maxRetries = 20; // Maximum number of retries
        int retryDelay = 1; // Delay between retries in seconds
        int currentRetry = 0;

        while (currentRetry < maxRetries)
        {
            try
            {
                if (isConnectedToPoseModel)
                {
                    UnityEngine.Debug.Log($"sunny test Attempting to connect to Python server... (Attempt {currentRetry + 1}/{maxRetries})");
                    client = new TcpClient(ADDRESS, PORT);
                    stream = client.GetStream();
                    stream.ReadTimeout = READ_TIME_OUT;
                    UnityEngine.Debug.Log("Connected to Python server");
                    player_movement.SetConnectedToSocket(true);
                    connectedToSocket = true;
                    yield break; // Successfully connected, exit coroutine
                }
                else
                {
                    UnityEngine.Debug.LogWarning($"sunny test Connection attempt failed as pose model is not connected.");
                    currentRetry++;
                }
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogWarning($"sunny test Connection attempt failed: {e.Message}");
                currentRetry++;
            }

            // Move the retry logic outside the catch block
            if (currentRetry < maxRetries)
            {
                // Wait before retrying - this is now outside the catch block
                yield return new WaitForSecondsRealtime(retryDelay);
            }
            else
            {
                UnityEngine.Debug.LogError("sunny test Failed to connect to Python server after multiple attempts.");

            #if UNITY_EDITOR
                UnityEditor.EditorApplication.isPlaying = false; // Stops Play Mode
            #else
            Application.Quit();
            #endif

                yield break;
            }
        }
    }

    void GetQuoteReadySignal()
    {
        while (!connectedToSocket);

        byte[] buffer = new byte[4];
        int bytes_read = 0;

        while (isReceivingData)
        {
            try
            {
                bytes_read = stream.Read(buffer, 0, buffer.Length);
                if (bytes_read == 0)
                {
                    isReceivingData = false;
                    DisconnectServer();
                }
                signal = BitConverter.ToInt32(buffer, 0);
            }
            catch (Exception e)
            {
                UnityMainThreadDispatcher.Instance().Enqueue(() => UnityEngine.Debug.LogError($"Error receiving data for emotion recognition: {e.Message}"));
            }
        }
    }

    private void OnApplicationQuit()
    {
        // Stop the thread
        isReceivingData = false;

        try
        {
            if (stream != null)
            {
                byte[] exit_message = Encoding.UTF8.GetBytes("exit");
                stream.Write(exit_message, 0, exit_message.Length);
                stream.Flush();
            }
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogError($"Error while sending termination signal: {e.Message}");
        }
        if (stream != null) stream.Close();
        if (client != null) client.Close();

        if (python_process != null && python_process.HasExited)
        {
            python_process.Kill();
            python_process.Dispose();
        }
    }

    public int GetSignal()
    {
        return signal;
    }

    public void ResetSignal()
    {
        signal = 0;
    }

    // This method is used to ensure the pose model is ready before connecting to emotion recognition model
    public void ConnectedToPoseModel()
    {
        isConnectedToPoseModel = true;
    }

    private void DisconnectServer()
    {
        UnityMainThreadDispatcher.Instance().Enqueue(() =>
        {
            UnityEngine.Debug.LogError("Server closed the connection or connection lost.");
        #if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false; // Stops Play Mode
        #else
            Application.Quit();
        #endif
        });
    }
}
