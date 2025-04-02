import requests
from dotenv import load_dotenv
import os
import json

load_dotenv()
mbd_api_key = os.getenv("MBD_API_KEY")

import requests

url = "https://api.mbd.xyz/v2/farcaster/users/feed/similar"
payload = {
    "user_id": "123",
    "top_k": 3,
}
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": f"Bearer {mbd_api_key}"
}
response = requests.post(url, json=payload, headers=headers)
print(json.loads(response.text))