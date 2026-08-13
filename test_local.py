import cv2
import chess_ai.gridfind as gridfind
import numpy as np

img = cv2.imread('/home/luongminhthu/.gemini/antigravity-ide/brain/6fb8a2bd-9bc5-42a5-84a1-d2b505c96b95/media__1786589270981.png')
# Crop the camera part roughly (the top left box)
cam = img[80:550, 30:520]
gray = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)
print("Sharpness:", gridfind.sharpness(gray))
cands = gridfind._quad_candidates(gray)
print("Cands count:", len(cands))
if len(cands) > 0:
    for i, c in enumerate(cands[:3]):
        sc, _ = gridfind.caro_score(gridfind.warp(gray, c, 160))
        print(f"Cand {i} score: {sc}")
res = gridfind.find_board(cam)
print("find_board returns:", "SUCCESS" if res else "NONE")

