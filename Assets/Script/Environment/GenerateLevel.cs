using UnityEngine;
using System.Collections;
using Unity.VisualScripting;
using UnityEditor;
using System.Collections.Generic;

public class GenerateLevel : MonoBehaviour
{
    public GameObject[] section;

    [SerializeField] private int SECTION_Z_POSITION_MUPLITER = 140; //each time section is placed in a multiper of 140
    [SerializeField] private int SECTION_Z_FLIP_OFFSET;
    [SerializeField] private float SPAWN_POSITION = 210f; // how close the player needs to be spawn the next section
    [SerializeField] private int START_SECTION_NUM = 5;
    [SerializeField] private int MAX_SECTION_NUM = 6;

    private Queue<GameObject> section_buffer = new Queue<GameObject>();
    [SerializeField] private int z_position = 0; //starting position of starting section
    [SerializeField] private Transform player; // reference to player
    [SerializeField] private int section_number = -1;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        SECTION_Z_FLIP_OFFSET = SECTION_Z_POSITION_MUPLITER;
        z_position = 0;

        // Spawn the 2 sections at the start
        for (int i = 0; i < START_SECTION_NUM; i++)
        {
            GenerateSection(true);
        }

    }

    // Update is called once per frame
    void Update()
    {
        if (player.position.z + SPAWN_POSITION > z_position)
        {
            GenerateSection();
            RemoveOldestSection();
        }
    }

    void GenerateSection(bool Start = false)
    {
        section_number = Random.Range(0, 3);

        // generate/clone whatever is in the section[section_number]
        GameObject new_section = Instantiate(section[section_number], new Vector3(0, 0, z_position), Quaternion.identity);
        if (Random.Range(0, 2) == 1) new_section.transform.localScale = new Vector3(-1, 1, 1);

        section_buffer.Enqueue(new_section);

        // the next z_position for next section
        z_position += SECTION_Z_POSITION_MUPLITER;
    }

    void RemoveOldestSection()
    {
        if (section_buffer.Count > MAX_SECTION_NUM)
        {
            // Remove the oldest section
            GameObject oldest_section = section_buffer.Dequeue(); // Remove from the queue
            Destroy(oldest_section); // Destroy the GameObject
        }
    }
}
