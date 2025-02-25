using TMPro;
using UnityEngine;
using UnityEngine.UI;

// Used for coins.
public class CollectableControl : MonoBehaviour
{
    [SerializeField] private GameObject coin_display;
    private static int coin_num = 0;

    // Update is called once per frame
    void Update()
    {
        coin_display.GetComponent<TextMeshProUGUI>().text = $"{coin_num}";
    }

    public static int GetCurrentCoin()
    {
        return coin_num; 
    }

    public static void IncreaseCoinByOne()
    {
        coin_num++;
    }

    public static void CoinToZero()
    {
        coin_num = 0;
    }
}
