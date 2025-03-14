using System.Collections.Generic;
using UnityEngine;
using System.Collections;
using UnityEngine.UIElements;

public class ExerciseChallengePlaneGeneration : MonoBehaviour
{
    [SerializeField] private GameObject exercise_challenge_plane_generation; // Must be initialized in inspector 

    [SerializeField] private float spawn_interval = 20f;
    [SerializeField] private float spawn_interval_mini = 15f;
    [SerializeField] private float spawn_interval_max = 40f;
    [SerializeField] private float spawn_distance = 140f; // Distance between character and dynamic obstacle while spawning

    private bool shouldStopCoroutine = false;

    [SerializeField] private GameObject player;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        StartCoroutine(SpawnPlane());
    }

    // Update is called once per frame
    IEnumerator SpawnPlane()
    {
        while (!shouldStopCoroutine)
        {
            //Debug.Log($"Spawning plane");

            float z = player.transform.position.z + spawn_distance; // Spawn in front of the character with distacne of spawn_distance by changing position.z
            Vector3 spawn_position = new Vector3(0, 0, z);

            GameObject new_plane = Instantiate(exercise_challenge_plane_generation, spawn_position, exercise_challenge_plane_generation.transform.rotation);

            //ObstacleRemoval class can also used by plane.
            new_plane.AddComponent<ObstacleRemoval>();

            yield return new WaitForSeconds(spawn_interval);

            // Ramdonly assign next spawning distance based on mini and max interval value.
            spawn_interval = Random.Range(spawn_interval_mini, spawn_interval_max);
        }
    }

    public void StopCoroutine()
    {
        shouldStopCoroutine = true;
    }

    private Transform CreatePlanePoint(Vector3 position)
    {
        GameObject spawn_point = new GameObject("SpawnPoint");
        spawn_point.transform.position = position;
        return spawn_point.transform;
    }
}


