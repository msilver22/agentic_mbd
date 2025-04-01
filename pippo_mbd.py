import requests
from dotenv import load_dotenv
import os

load_dotenv()
mbd_api_key = os.getenv("MBD_API_KEY")

import requests

url = "https://api.mbd.xyz/v2/farcaster/casts/search/semantic"

payload = {
    "query": "Trump is a good person?",
    "return_metadata": True
}
headers = {
    "accept": "application/json",
    "HTTP-Referer": "https://docs.mbd.xyz/",
    "X-Title": "mbd_docs",
    "content-type": "application/json",
    "authorization": f"Bearer {mbd_api_key}"
}

response = requests.post(url, json=payload, headers=headers)

print(response.text)