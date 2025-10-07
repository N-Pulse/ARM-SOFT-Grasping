from avp_stream import VisionProStreamer
s = VisionProStreamer(ip="IP_DU_CASQUE", record=False)
while True:
    r = s.latest
    T = r["right_wrist"][0]
    print("right_wrist[0,0..2,0..2] =", T[:3,:3])  
