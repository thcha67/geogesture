import keyboard
import mouse
import random
import time


OPEN_MAP = (-486, 815)

OPENED_MAP_CORNERS = [
    (-1300, 500),
    (-400, 500),
    (-1300, 850),
    (-400, 850),
]

GUESS = (-500, 920)

NEXT_ROUND = GUESS

SCREEN_SIZE = (1920, 1080)

SCREEN_CENTER = (SCREEN_SIZE[0] // 2, SCREEN_SIZE[1] // 2)


#mouse.move(*SCREEN_CENTER, absolute=True, duration=0.1)

keyboard.press_and_release("haut")
keyboard.press_and_release("haut")
keyboard.press_and_release("gauche")
keyboard.press_and_release("gauche")



def move():
    mouse.move(*OPEN_MAP, absolute=True, duration=0.1)

    mouse.move((OPENED_MAP_CORNERS[0][0] + OPENED_MAP_CORNERS[1][0]) // 2, (OPENED_MAP_CORNERS[0][1] + OPENED_MAP_CORNERS[2][1]) // 2, absolute=True, duration=1)

    for _ in range(100):
        time.sleep(0.01)
        mouse.wheel(-120)

    for _ in range(20):
        time.sleep(0.01)
        mouse.wheel(120)

    for corner in OPENED_MAP_CORNERS:
        mouse.move(*corner, absolute=True, duration=0.1)

    random_guess_within_corners = (
        random.randint(OPENED_MAP_CORNERS[0][0], OPENED_MAP_CORNERS[1][0]),
        random.randint(OPENED_MAP_CORNERS[0][1], OPENED_MAP_CORNERS[2][1])
    )

    mouse.move(*random_guess_within_corners, absolute=True, duration=1)
    mouse.click()

    mouse.move(*GUESS, absolute=True, duration=1)
    mouse.click()

    time.sleep(1)

    mouse.move(*NEXT_ROUND, absolute=True, duration=1)
    mouse.click()

