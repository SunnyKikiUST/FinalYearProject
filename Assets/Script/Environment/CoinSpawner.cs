using System.Collections.Generic;
using UnityEngine;
using System.Collections;

public class CoinSpawner : MonoBehaviour
{
    [SerializeField] private GameObject coin; // Must be initialized in inspector 

    [SerializeField] private Transform[] spawn_points = new Transform[3]; // Assign 3 spawn points
    [SerializeField] private float spawn_interval = 0.5f;
    [SerializeField] private float spawn_distance = 140f; // Distance between character and dynamic obstacle while spawning
    [SerializeField] private float min_restrict_distance = 5f; 

    [SerializeField] private float left_path_x = -3.75f;
    [SerializeField] private float middle_path_x = 0f;
    [SerializeField] private float right_path_x = 3.75f;

    private bool shouldStopCoroutine = false;

    [SerializeField] private GameObject player;
    [SerializeField] private GameObject character_model;

    private List<GameObject> obstacles = null;
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        obstacles = GetComponent<ObstacleSpawner>().GetObstacles();

        Debug.Log($"CoinSpawner Starts");
        // Create spawn points 
        spawn_points[0] = CreateSpawnPoint(new Vector3(left_path_x, 0f, 0f));
        spawn_points[1] = CreateSpawnPoint(new Vector3(middle_path_x, 0f, 0f));
        spawn_points[2] = CreateSpawnPoint(new Vector3(right_path_x, 0f, 0f));

        StartCoroutine(SpawnCoin());
    }

    // Update is called once per frame
    IEnumerator SpawnCoin()
    {
        while (!shouldStopCoroutine)
        {
            Debug.Log($"Sppawning Coin");
            int random_lane = Random.Range(0, spawn_points.Length);

            Vector3 spawn_position = spawn_points[random_lane].position;
            spawn_position.z = player.transform.position.z + spawn_distance; // Spawn in front of the character with distacne of spawn_distance by changing position.z
            spawn_position.y = 1f;


            foreach (GameObject obstacle in obstacles)
            {
                if (obstacle != null && Vector3.Distance(obstacle.transform.position, spawn_position) < min_restrict_distance)
                {
                    Debug.Log("Dynamic obstacle cannot spawn due to min_restrict_distance.");
                    continue;
                }
            }

            GameObject new_coin = Instantiate(coin, spawn_position, coin.transform.rotation);

            yield return new WaitForSeconds(spawn_interval);
        }
    }

    public void StopCoroutine()
    {
        shouldStopCoroutine = true;
    }

    private Transform CreateSpawnPoint(Vector3 position)
    {
        GameObject spawn_point = new GameObject("SpawnPoint");
        spawn_point.transform.position = position;
        return spawn_point.transform;
    }
}


