import cv2
import numpy as np
from pynput.keyboard import Key, Controller
from PIL import ImageGrab
from matplotlib import pyplot as plt

jump_dist = 160

keyboard = Controller()



dinoTemplate = cv2.imread('./obj_templates/dino_s1.png',0)
dw, dh = dinoTemplate.shape[::-1]
 
cactusTemplate = cv2.imread('./obj_templates/cactus_big.png',0)
cw, ch = cactusTemplate.shape[::-1]

cactusMinTemplate = cv2.imread('./obj_templates/cactus_min_1.png',0)
cmw, cmh = cactusMinTemplate.shape[::-1]

overTemplate = cv2.imread('./obj_templates/over.png',0)
ow, oh = overTemplate.shape[::-1]

threshold = 0.7


while(True):
    screen = np.array(ImageGrab.grab(bbox=(865,315,1321,520)))
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

    dinoRes = cv2.matchTemplate(screen_gray, dinoTemplate, cv2.TM_CCOEFF_NORMED)
    dinoLoc = np.where(dinoRes >= threshold)

    cactusRes = cv2.matchTemplate(screen_gray, cactusTemplate, cv2.TM_CCOEFF_NORMED)
    cactusLoc = np.where(cactusRes >= threshold)

    cactusMinRes = cv2.matchTemplate(screen_gray, cactusMinTemplate, cv2.TM_CCOEFF_NORMED)
    cactusMinLoc = np.where(cactusMinRes >= threshold)

    overRes = cv2.matchTemplate(screen_gray, overTemplate, cv2.TM_CCOEFF_NORMED)
    overLoc = np.where(overRes >= threshold)

    for over_pt in zip(*overLoc[::-1]):
        keyboard.press(Key.space)
        break

    for dino_pt in zip(*dinoLoc[::-1]):
        cv2.rectangle(screen, dino_pt, (dino_pt[0] + dw, dino_pt[1] + dh), (0,0,0), 1)
        dino_end = dino_pt[1]
        break

    for cactus_pt in zip(*cactusLoc[::-1]):
        cv2.rectangle(screen, cactus_pt, (cactus_pt[0] + cw, cactus_pt[1] + ch), (0,0,255), 1)
        cactus_start = cactus_pt[0]
        if((cactus_start - dino_end) < jump_dist):
            keyboard.press(Key.space)
        break

    for cactus_min_pt in zip(*cactusMinLoc[::-1]):
        cv2.rectangle(screen, cactus_min_pt, (cactus_min_pt[0] + cmw, cactus_min_pt[1] + cmh), (0,0,255), 1)
        cactus_min_start = cactus_min_pt[0]
        if((cactus_min_start - dino_end) < jump_dist):
            keyboard.press(Key.space)
        break

    cv2.imshow('output.png', screen)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break