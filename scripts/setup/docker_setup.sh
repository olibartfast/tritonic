#!/bin/bash

# This script automates the installation of Docker and the NVIDIA Container Toolkit on Ubuntu.
# It follows the official installation guides to ensure a correct setup.
#
# References:
# - Docker: https://docs.docker.com/engine/install/ubuntu/
# - NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Installing Docker ---
echo "--- Step 1: Installing Docker ---"

echo "[Docker] Setting up Docker's apt repository..."
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo "[Docker] Adding Docker's repository to apt sources..."
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

echo "[Docker] Installing Docker packages..."
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "[Docker] Adding the current user to the Docker group..."
sudo usermod -aG docker $USER
echo "IMPORTANT: For the group change to take effect, you may need to log out and log back in, or run 'newgrp docker' in your shell."

echo "--- Docker installation completed. ---"
echo ""

# --- Installing the NVIDIA Container Toolkit ---
echo "--- Step 2: Installing the NVIDIA Container Toolkit ---"

echo "[NVIDIA] Configuring the production repository..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update

echo "[NVIDIA] Installing the NVIDIA Container Toolkit packages..."
sudo apt-get install -y nvidia-container-toolkit

echo "[NVIDIA] Configuring Docker to use the NVIDIA runtime..."
sudo nvidia-ctk runtime configure --runtime=docker

echo "--- NVIDIA Container Toolkit installation completed. ---"
echo ""

# --- Testing Docker and NVIDIA Container Runtime ---
echo "--- Step 3: Testing the installation ---"
echo "A system restart might be required for all changes to take effect."
echo "Attempting to restart the Docker service..."
sudo systemctl restart docker

echo "[Test] Running a sample CUDA container to verify the setup..."
echo "The following command will run nvidia-smi inside a container. You should see your GPU details."

sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi

echo ""
echo "--- Installation and Test Script Finished. ---"
echo "If the nvidia-smi command above showed your GPU information, the installation was successful."
echo "If you encountered any errors, please consult the official documentation linked at the top of the script."
