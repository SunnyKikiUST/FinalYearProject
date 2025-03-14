using UnityEngine;

public class PauseInGameMoment : MonoBehaviour
{

    //private GameObject player;
    //private CoinSpawner coin_spawner;
    //private ObstacleSpawner obstacle_spawner;
    //private LevelScore level_score;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        //GameObject level_control = GameObject.Find("LevelControl");
        //player = GameObject.Find("Player");
        //coin_spawner = level_control.GetComponent<CoinSpawner>();
        //obstacle_spawner = level_control.GetComponent<ObstacleSpawner>();
        //level_score = level_control.GetComponent <LevelScore>();
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKey(KeyCode.Q))
        {
            Debug.Log("StopInGame StopInGameState");
            PauseInGameState();
        }
        else if (Input.GetKey(KeyCode.E))
        {
            Debug.Log("StopInGame StartInGameState");
            ResumeInGameState();
        }
    }



    // Below is the thing need to be stopped:
        // Player Movement 
        // Coin Spawner 
        // Obstacle Spawner 
        // Moving of Dynamic Obstacle
        // Distance still counting
    public void PauseInGameState()
    {
        Time.timeScale = 0; //Freeze Time.
        //player.GetComponent<PlayerMovement>().PauseMovementState();
    }

    public void ResumeInGameState()
    {
        Time.timeScale = 1; //Resume Time
        //player.GetComponent<PlayerMovement>().ResumeMovementState();
    }
}
