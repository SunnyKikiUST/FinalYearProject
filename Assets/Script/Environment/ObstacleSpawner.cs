using UnityEngine;
using System.Collections;

using System.Collections.Generic;
using MongoDB.Driver.Core.Misc;

public class ObstacleSpawner : MonoBehaviour
{
    [SerializeField] private GameObject[] static_obstacle_prefabs; // Must be initialized in inspector 
    [SerializeField] private GameObject[] dynamic_obstacle_prefabs; // Must be initialized in inspector 

    [SerializeField] private Transform[] spawn_points = new Transform[3]; // Assign 3 spawn points
    [SerializeField] private float spawn_interval = 0.1f;
    [SerializeField] private float dynamic_obstacle_speed = 10f;
    [SerializeField] private float spawn_distance = 140f; // Distance between character and dynamic obstacle while spawning
    [SerializeField] private float min_allow_distance = 70f; // Min distance that is not allowed between spawning dynamic obstacle and static obstacle in the same line
    [SerializeField] private float min_allow_distance_dy = 35f; // Min distance that is not allowed between 2 dynamic obstacles in the same line
    [SerializeField] private float min_allow_distance_st = 35f; // Min distance that is not allowed between 2 static obstacles in the same line

    [SerializeField] private float left_path_x = -3.75f;
    [SerializeField] private float middle_path_x = 0f;
    [SerializeField] private float right_path_x = 3.75f;

    [SerializeField] private GameObject player;
    [SerializeField] private GameObject character_model;

    private bool shouldStopCoroutine = false;

    private int previous_dy_obstacle_index = -1;
    private int previous_st_obstacle_index = -1;

    private List<GameObject> static_obstacles = new List<GameObject>();
    private List<GameObject> dynamic_obstacles = new List<GameObject>();
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        Debug.Log($"ObstacleSpawner Starts");
        // Create spawn points 
        spawn_points[0] = CreateSpawnPoint(new Vector3(left_path_x, 0f, 0f));
        spawn_points[1] = CreateSpawnPoint(new Vector3(middle_path_x, 0f, 0f));
        spawn_points[2] = CreateSpawnPoint(new Vector3(right_path_x, 0f, 0f));

        StartCoroutine(SpawnObstacles());
    }

    private void Update()
    {
        static_obstacles.RemoveAll(item => item == null);
        dynamic_obstacles.RemoveAll(item => item == null);
    }

    // Update is called once per frame
    IEnumerator SpawnObstacles()
    {
        Debug.Log($"ObstacleSpawner is updating once per frame");
        Debug.Log($"Player is {player}");

        //Ensure that spawning dynamic obstacle must be successful
        bool isDynamicSuccessful = true;

        while (!shouldStopCoroutine)
        {
            int spawn_number = Random.Range(1, 3);
            int random_lane = Random.Range(0, spawn_points.Length);

            int random_type = 0;

            if (isDynamicSuccessful)
            {
                random_type = Random.Range(0, 2);
            }
            // Try again spawning dynamic obstacle if last time is not successful.
            else
            {
                random_type = 1;
            }

            for (int i = 0; i < spawn_number; i++)
            {

                // Spawn static obstacle
                if (random_type == 0)
                {
                    int length = static_obstacle_prefabs.Length;
                    int random_obstacle = Random.Range(0, length);
                    if (random_obstacle == previous_st_obstacle_index)
                    {
                        random_obstacle = (random_obstacle++) % length;
                    }
                    previous_st_obstacle_index = random_obstacle;
                    SpawnStaticObstacle(random_lane, random_obstacle);
                }
                // Spawn dynamic obstacle
                else
                {
                    int random_obstacle = Random.Range(0, dynamic_obstacle_prefabs.Length);

                    //Ensure that spawning dynamic obstacle must be successful
                    isDynamicSuccessful = SpawnDynamicObstacle(random_lane, random_obstacle);
                }

                // Increase chance of getting 2 obstacle in same distance but in different paths
                random_lane = (random_lane + 1) % 2;
            }

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


    // Spawn static obstacle
    void SpawnStaticObstacle(int lane_index, int random_obstacle)
    {
        Debug.Log($"Inside SpawnStaticObstacle");
        Vector3 spawn_position = spawn_points[lane_index].position;
        spawn_position.z = player.transform.position.z + spawn_distance; // Spawn in front of the character with distacne of spawn_distance by changing position.z

        foreach (GameObject obstacle in static_obstacles)
        {
            if (obstacle != null && 
                Mathf.Abs(obstacle.transform.position.z - spawn_position.z) < min_allow_distance_st && 
                spawn_position.x == obstacle.transform.position.x)
            {
                Debug.Log("Static obstacle cannot spawn due to min_allow_distance_st.");
                return;
            }
        }

        GameObject selected_object = static_obstacle_prefabs[random_obstacle];
        GameObject static_obstacle = Instantiate(static_obstacle_prefabs[random_obstacle], spawn_position, selected_object.transform.rotation);

        static_obstacle.AddComponent<ObstacleRemoval>();

        // Instead of attaching ObstacleCollision here, I have manually attached to desired gameobject, as it is not always the parent gameobject need the script.
        //ObstacleCollision ObstacleCollisionClass = static_obstacle.AddComponent<ObstacleCollision>();

        static_obstacles.Add(static_obstacle);
        Debug.Log($"Static obstacle spawned at: {spawn_position}");
    }

    // Spawn dynamic obstacle
    bool SpawnDynamicObstacle(int lane_index, int random_obstacle)
    {
        Debug.Log($"Inside SpawnDynamicObstacle");
        Vector3 lane_position = spawn_points[lane_index].position;
        Vector3 spawn_position = lane_position + Vector3.forward * spawn_distance;
        spawn_position.z = player.transform.position.z + spawn_distance; // Spawn in front of the character with distacne of spawn_distance by changing position.z

        foreach (GameObject obstacle in static_obstacles)
        {
            if (obstacle != null && Vector3.Distance(obstacle.transform.position, spawn_position) < min_allow_distance)
            {
                Debug.Log("Dynamic obstacle cannot spawn due to min_allow_distance.");
                return false;
            }
        }

        foreach (GameObject obstacle in dynamic_obstacles)
        {
            if (obstacle != null &&
                Mathf.Abs(obstacle.transform.position.z - spawn_position.z) < min_allow_distance_dy &&
                spawn_position.x == obstacle.transform.position.x)
            {
                Debug.Log("Static obstacle cannot spawn due to min_allow_distance_st.");
                return false; ;
            }
        }

        GameObject selected_object = dynamic_obstacle_prefabs[random_obstacle];
        GameObject dynamic_obstacle = Instantiate(selected_object, spawn_position, selected_object.transform.rotation);

        dynamic_obstacle.AddComponent<MovingObstacleMotion>().StartMoving();
        dynamic_obstacle.AddComponent<ObstacleRemoval>();

        // Instead of attaching ObstacleCollision here, I have manually attached to desired gameobject, as it is not always the parent gameobject need the script.
        //ObstacleCollision ObstacleCollisionClass = dynamic_obstacle.AddComponent<ObstacleCollision>();
        dynamic_obstacles.Add(dynamic_obstacle);
        Debug.Log($"Dynamic obstacle spawned at: {spawn_position}");

        return true;
    }

    public List<GameObject> GetStaticObstacles()
    {
        return static_obstacles;
    }

    public List<GameObject> GetDynamicObstacles()
    {
        return dynamic_obstacles;
    }


}
