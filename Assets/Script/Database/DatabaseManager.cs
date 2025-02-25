using UnityEngine;
using MongoDB.Driver;
using MongoDB.Bson;
using MongoDB.Bson.Serialization.Attributes;
using System.Security.Cryptography;
using System;
using System.Threading.Tasks;
using UnityEngine.SocialPlatforms.Impl;
using System.Drawing;

public class DatabaseManager : MonoBehaviour
{
    // Holds the single instance for our Singleton implementation
    public static DatabaseManager Instance;

    private MongoClient client;
    private IMongoDatabase database;
    private IMongoCollection<User> usersCollection;
    private IMongoCollection<Record> recordsCollection;

    private const string connectionUri = "mongodb+srv://ykslam:HKUST@hkust2025fypmxj1.bthod.mongodb.net/?retryWrites=true&w=majority&appName=HKUST2025FYPMXJ1";

    private const int SaltSize = 16; // 128-bit salt
    private const int KeySize = 32;  // 256-bit key
    private const int Iterations = 10000;

    private string loginCollactionName = "Users";
    private string leaderboardCollectionName = "HighScoreAndCoinsCollection";

    public class User
    {
        [BsonId]
        private ObjectId Id { get; set; }

        private string user_name { get; set; }
        private string hashed_password { get; set; }

        public User(string user_name, string hashed_password)
        {
            this.user_name = user_name;
            this.hashed_password = hashed_password;
        }

        public string GetUserName() {  return user_name; }
        public string GetHashedPassword() {  return hashed_password; }
    }

    public class Record
    {
        [BsonId]
        private ObjectId Id { get; set; }
        private string user_name { get; set; }
        private int coins { get; set; }
        private int score { get; set; }
        public int point { get; set; }
        public Record(string user_name)
        {
            this.user_name= user_name;
            this.coins = 0;
            this.score = 0;
            this.point = 0;
        }

        public string GetUserName() { return user_name; }
        public int GetCoins() { return coins; }
        public int GetScore() { return score; }
        public int GetPoint() { return point; }

    }


    void Awake()
    {
        // Initialize the singleton instance. If one exists, ensure only one exists.
        if (Instance == null)
        {
            Instance = this;

            //IMPORTANT
            DontDestroyOnLoad(gameObject);  // Keeps this object when switching scenes. In addition, such operation will ensure the parent gameObject and the attached component (like other scripts) not to destroy also. 
        }
        else
        {
            Destroy(gameObject);
        }
    }

    void Start()
    {
        var settings = MongoClientSettings.FromConnectionString(connectionUri);
        settings.ServerApi = new ServerApi(ServerApiVersion.V1);
        client = new MongoClient(settings);

        database = client.GetDatabase("GameDB");
        usersCollection = database.GetCollection<User>(loginCollactionName);
        recordsCollection = database.GetCollection<Record>(leaderboardCollectionName);
    }

    // Get password of the player
    private async Task<string> GetActualPassword(string userName)
    {
        try
        {
            User retrievedUser = await usersCollection.Find(info => info.GetUserName() == userName).FirstOrDefaultAsync();
            if (retrievedUser != null) return retrievedUser.GetHashedPassword();
            else return null;
        }
        catch (Exception ex)
        {
            return "";
        }
    }

    // Get Record of the player
    public async Task<Record> GetRecord(string userName)
    {
        try
        {
            Record retrievedRecord = await recordsCollection.Find(info => info.GetUserName() == userName).FirstOrDefaultAsync();
            if (retrievedRecord != null) return retrievedRecord;
            else return null;
        }
        catch (Exception ex) {
            return null;
        }
    }

    // Check whether the player exist in database
    private async Task<int> IsUserExist(string userName)
    {
        try
        {
            var projection = Builders<User>.Projection.Include(info => info.GetUserName());
            User retrievedUser = await usersCollection.Find(info => info.GetUserName() == userName).Project<User>(projection).FirstOrDefaultAsync();
            if (retrievedUser != null) return 1;
            else return 0;
        }
        catch (Exception ex) {
            return -1;
        }
    }

    // Register new player and the password
    private async Task<bool> StoreUserAndPassword(string userName, string password)
    {
        try
        {
            User new_info = new User(userName, password);
            await usersCollection.InsertOneAsync(new_info);
            return true;
        }
        catch (Exception ex) {
            return false;
        }
    }
    
    //After register a new player, assign the player a initial record;
    private async Task<bool> StoreInitialRecord(string userName)
    {
        try
        {
            Record new_record = new Record(userName);
            await recordsCollection.InsertOneAsync(new_record);
            return true;
        }
        catch (Exception ex)
        {
            return false;
        }
    }

    // This method authenticates a player by checking if a document matching the username and password exists.
    // userAttemptedPassword: the password that player input and want to login
    public async Task<string> AuthenticateUser(string userName, string userAttemptedPassword)
    {
        // Return false if no such user exist.
        string hashedPassword = await GetActualPassword(userName);
        if (hashedPassword == null) return "Invalid username or password.";
        else if (hashedPassword == "") return "Error happened while accessing to database. Please try again.";

        // Convert the Base64 string back to bytes.
        var hashBytes = Convert.FromBase64String(hashedPassword);
        if (hashBytes.Length != SaltSize + KeySize)
            return "Invalid username or password.";

        // Extract the salt from the stored hash. This is for generating a key again in order to have comparison.
        var salt = new byte[SaltSize];
        Buffer.BlockCopy(hashBytes, 0, salt, 0, SaltSize);

        // Derive the key from the provided password using the extracted salt.
        byte[] keyToCheck;
        using (var pbkdf2 = new Rfc2898DeriveBytes(userAttemptedPassword, salt, Iterations, HashAlgorithmName.SHA256))
        {
            keyToCheck = pbkdf2.GetBytes(KeySize);
        }

        // Compare the newly derived key with the stored key.
        for (int i = 0; i < KeySize; i++)
        {
            if (hashBytes[i + SaltSize] != keyToCheck[i])
                return "Invalid username or password.";
        }
        return "success";
    }

    public async Task<string> RegisterUser(string userName, string userPassword)
    {
        int result = await IsUserExist(userName);
        // Return false if user exist.
        if (result == 1) return "The user name has been registered.";
        else if (result == -1) return "Error happened while accessing to database. Please try again.";

        byte[] salt;
        // Generate a cryptographically secure random salt.
        using (var rng = RandomNumberGenerator.Create())
        {
            salt = new byte[SaltSize];
            rng.GetBytes(salt);
        }

        // Use PBKDF2 (Password-Based Key Derivation Function 2) to hash the password along with the salt.
        byte[] key;
        using (var pbkdf2 = new Rfc2898DeriveBytes(userPassword, salt, Iterations, HashAlgorithmName.SHA256))
        {
            key = pbkdf2.GetBytes(KeySize);
        }

        // Combine the salt and the derived key into one byte array for storage.
        var hashBytes = new byte[SaltSize + KeySize];
        Buffer.BlockCopy(salt, 0, hashBytes, 0, SaltSize);
        Buffer.BlockCopy(key, 0, hashBytes, SaltSize, KeySize);

        // Convert the combined byte array (salt + key) to a Base64 string for easy storage.
        if (!await StoreUserAndPassword(userName, Convert.ToBase64String(hashBytes))) return "Error happened while accessing to database. Please try again.";
        // Store initial record data to database.
        if (!await StoreInitialRecord(userName)) return "Error happened while accessing to database. Please try again.";
        return "Register successful!";

    }

    public async Task<bool> UpdateRecord(string userName, int coins, int score, int point)
    {
        try
        {
            // Create a filter to find the record document for the specified player name.
            var filter = Builders<Record>.Filter.Eq(r => r.GetUserName(), userName);

            // Create an update definition to set the coins and score.
            var update = Builders<Record>.Update
                   .Set(r => r.GetCoins(), coins)
                    .Set(r => r.GetScore(), score)
                    .Set(r => r.GetPoint(), point);

            // Execute the update on the collection.
            var result = await recordsCollection.UpdateOneAsync(filter, update);

            // Return true if one document was modified.
            return result.ModifiedCount > 0;
        }
        catch (Exception ex)
        {
            // Logging error details can help diagnose issues.
            Debug.LogError($"Error updating record for user {userName}: {ex.Message}");
            return false;
        }
    }
}