#petit logger pour générer des traces de main/poignet, utiles pour tester le pont et pour l’IL plus tard
import json, time
from avp_stream import VisionProStreamer

s = VisionProStreamer(ip="IP_DU_CASQUE", record=False)
with open("vp_stream.jsonl","w") as f:
    while True:
        r = s.latest
        r["t"] = time.time()
        f.write(json.dumps(r) + "\n")
