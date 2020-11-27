try:
    import RPi.GPIO as GPIO
except ImportError as e:
    print(f"[ERROR] Deploy this program on Raspberry Pi ({e})")
    quit()

GPIO.setmode(GPIO.BCM)


class ObstacleDetector:
    def __init__(self, pin=4):
        self.pin = pin
        GPIO.setup(self.pin, GPIO.IN)

    def get_obstacle_data(self):
        state = GPIO.input(self.pin)
        return int(not state)

if __name__ == '__main__':
    a = ObstacleDetector()
    print(a.get_obstacle_data())
