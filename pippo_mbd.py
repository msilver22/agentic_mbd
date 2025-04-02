import requests
from dotenv import load_dotenv
import os
import json

load_dotenv()
mbd_api_key = os.getenv("MBD_API_KEY")

import requests

url = "https://api.mbd.xyz/v2/farcaster/casts/feed/trending"
#payload = {
#    "query": "war in Israel",
#    "top_k": 3,
#    "return_metadata": True,
#}
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": f"Bearer {mbd_api_key}"
}
#response = requests.post(url, json=payload, headers=headers)
response = requests.post(url,headers=headers)
print(json.loads(response.text))