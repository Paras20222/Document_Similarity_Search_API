import os
import sys
import pandas as pd
import aiohttp
import asyncio
import json

API_URL = "http://localhost:8000/api/documents"
LOCAL_DATASET_PATH = r"C:\Users\paras\OneDrive\Desktop\AG_NEWS_DATASET.csv"
CHECKPOINT_FILE = "last_uploaded_index.txt"
BATCH_SIZE = 500  # Increased batch size for fewer API calls
CONCURRENT_REQUESTS = 100  # Adjust this to match API rate limits

def get_last_uploaded_index():
    return int(open(CHECKPOINT_FILE).read().strip()) if os.path.exists(CHECKPOINT_FILE) else 0

def save_last_uploaded_index(index):
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(index))

async def upload_document(session, document):
    try:
        async with session.post(API_URL, json=document) as response:
            if response.status == 200:
                return True
    except Exception as e:
        print(f"Upload failed: {e}")
    return False

async def upload_batch(session, batch):
    tasks = [upload_document(session, doc) for doc in batch]
    results = await asyncio.gather(*tasks)
    return sum(results)

async def main():
    if not os.path.exists(LOCAL_DATASET_PATH):
        sys.exit(1)

    last_uploaded = get_last_uploaded_index()
    total_uploaded = last_uploaded

    df = pd.read_csv(LOCAL_DATASET_PATH, chunksize=BATCH_SIZE)
    async with aiohttp.ClientSession() as session:
        for chunk in df:
            if total_uploaded >= len(chunk):
                continue

            batch = [
                {
                    "title": row["Title"],
                    "content": row["Description"],
                    "metadata": {
                        "category": row["Class Index"],
                        "source": "AG News Dataset",
                        "index": idx
                    }
                }
                for idx, row in chunk.iterrows()
            ]

            uploaded = await upload_batch(session, batch)
            total_uploaded += uploaded
            save_last_uploaded_index(total_uploaded)
            print(f"Uploaded {total_uploaded}/{7600} documents...")

    print("Upload complete!")

if __name__ == "__main__":
    asyncio.run(main())

