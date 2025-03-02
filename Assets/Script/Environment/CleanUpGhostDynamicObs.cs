using System.Collections.Generic;
using UnityEngine;

public class CleanUpGhostDynamicObs : MonoBehaviour
{
    // Assign your list in the inspector or populate it dynamically in code
    private List<GameObject> objectList;

    void Update()
    {
        // This will remove all null references from the list each frame.
        objectList.RemoveAll(item => item == null);
    }
}
