On the Raspberry Pi, start the camera web server
tmux attach -t rpi4-arduino
python3 ~/rpi4_camera.py

On the main Linux PC, activate your environment and start CLIP
cd ~/Desktop/UTsukuba/2025 Thesis/ManipulaTHOR-RL/rpi4-vision-ai-dashboard
conda activate ubuntu-rpi4
python pc_rpi4_camera.py

On the Raspberry Pi, start the Arduino LED bridge
ls /dev/ttyACM0 (Make sure the Arduino Uno is connected)
tmux attach -t rpi4-arduino
python3 ~/arduino/rpi4_arduino_led.py

Open the camera page from your PC browser
http://192.168.0.26:5000

From your Linux PC, use scp to copy both files from the Raspberry Pi.
scp ubuntu-rpi4@192.168.0.26:~/rpi4_camera.py \
    "$HOME/Desktop/UTsukuba/2025 Thesis/ManipulaTHOR-RL/rpi4-vision-ai-dashboard/"

scp ubuntu-rpi4@192.168.0.26:~/arduino/rpi4_arduino_led.py \
    "$HOME/Desktop/UTsukuba/2025 Thesis/ManipulaTHOR-RL/rpi4-vision-ai-dashboard/arduino/clip_led/"

SSH Commands
ping 192.168.0.26
ssh ubuntu-rpi4@192.168.0.26
pass: ubuntu-rpi4

ping 192.168.0.16
ssh muhamuttaqien@192.168.0.16
sudo systemctl status ssh

hostname -I
pkill -f pc_rpi4_camera.py

ChatGPT Raspberry Pi 4 & Arduino UNO
https://chatgpt.com/c/df813f51-a444-8331-868b-49e3fa93a63e
