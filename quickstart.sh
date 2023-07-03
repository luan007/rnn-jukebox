
sudo apt update
sudo apt install -y build-essential
sudo apt install -y git alsa-tools alsa-utils
sudo apt install -y fluidsynth


curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -
sudo apt-get install -y nodejs


sudo usermod -a -G audio $USER
