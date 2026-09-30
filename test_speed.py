import urllib.request
import json
import time

base_url = "http://127.0.0.1:8000"

t0 = time.time()
print("1. Selecting Indian Pines...")
req = urllib.request.Request(
    f"{base_url}/api/session/select_dataset",
    data=json.dumps({"dataset_name": "Indian Pines", "seed": 42}).encode(),
    headers={"Content-Type": "application/json"}
)
res = json.loads(urllib.request.urlopen(req).read().decode())
print(f"   Indian Pines selected in {time.time()-t0:.3f}s. Mode: {res['mode']}, Labeled: {res['num_labeled']}")

t1 = time.time()
print("2. Selecting Salinas...")
req2 = urllib.request.Request(
    f"{base_url}/api/session/select_dataset",
    data=json.dumps({"dataset_name": "Salinas", "seed": 42}).encode(),
    headers={"Content-Type": "application/json"}
)
res2 = json.loads(urllib.request.urlopen(req2).read().decode())
print(f"   Salinas selected in {time.time()-t1:.3f}s. Labeled: {res2['num_labeled']}")

t2 = time.time()
print("3. Switching back to Indian Pines...")
req3 = urllib.request.Request(
    f"{base_url}/api/session/select_dataset",
    data=json.dumps({"dataset_name": "Indian Pines", "seed": 42}).encode(),
    headers={"Content-Type": "application/json"}
)
res3 = json.loads(urllib.request.urlopen(req3).read().decode())
print(f"   Indian Pines switched back in {time.time()-t2:.3f}s.")
