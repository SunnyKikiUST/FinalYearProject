using TMPro;
using UnityEngine;

public class CurrentCoinAndScore : MonoBehaviour
{
    // The "highest coin and highest score is not techically the highest", instead the point that they combine, i.e. highest_coin + higest_score is the highest.
    private static int highest_coin = 0;
    private static int highest_score = 0;
    private static int best_point = 0;
    private static string user_name = "";

    // Get the player record from database for UI display and comparison of game result
    public static void RetrieveRecordFromDataBase(string userName)
    {
        DatabaseManager.Record record = DatabaseManager.Instance.GetRecord(userName);
        highest_coin = record.GetCoins();
        highest_score = record.GetScore();
        best_point = record.GetPoint();
        user_name = record.GetUserName();   
    }

    // Update record if the new point is higher than the best point after finishing one round of game.
    public static async void UpdateRecordToDataBase(int coin, int score, int point)
    {
        if (best_point < point) await DatabaseManager.Instance.UpdateRecordAsync(user_name, coin, score, point);
        highest_coin = coin;
        highest_score = score;
        best_point = point;
    }

    public static string GetUserName()
    {
        return user_name;
    }


    public static int GetHighestCoin()
    {
        return highest_coin;
    }

    public static int GetHighestScore() 
    { 
        return highest_score; 
    }
    public static int GetBestPoint()
    {
        return best_point;
    }
}
