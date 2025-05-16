
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import certifi

uri = "mongodb+srv://nisha:nisha@ragcluster.6zcesdr.mongodb.net/?retryWrites=true&w=majority&appName=RagCluster"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())

db = client.RAG_db
collection = db["user_db"]

