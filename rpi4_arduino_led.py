import time
import json
import serial
import urllib.request

CLIP_URL = "http://192.168.0.16:5000/clip"
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

arduino = serial.Serial(
    SERIAL_PORT,
    BAUD_RATE,
    timeout=1
)

# Arduino Uno usually resets when serial is opened
time.sleep(2)

last_state = None

print("CLIP -> Arduino LED started")

while True:
    try:
        with urllib.request.urlopen(CLIP_URL, timeout=2) as response:
            data = json.load(response)

        if not data.get("scores"):
            time.sleep(0.5)
            continue

        best = data["best"]
        score = data["scores"][0]["score"]

        person_detected = (
            best == "a person"
            and score >= 0.30
        )

        if person_detected != last_state:

            if person_detected:
                arduino.write(b"1")
                print(
                    f"LED ON  | {best} | {score:.3f}"
                )
            else:
                arduino.write(b"0")
                print(
                    f"LED OFF | {best} | {score:.3f}"
                )

            last_state = person_detected

    except Exception as e:
        print("Error:", e)

    time.sleep(0.5)
