import pandas as pd
from pymongo import MongoClient

CSV_FILE = "finaldata.csv"
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "pre_delinquency_db"
COLLECTION_NAME = "customer_features"

df = pd.read_csv(CSV_FILE)

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

records = df.to_dict(orient="records")

collection.delete_many({})  
collection.insert_many(records)

print(f"{len(records)} records inserted into MongoDB successfully.")