
## Installing Docker  
Reference: [Official Docker Installation Guide for Ubuntu](https://docs.docker.com/engine/install/ubuntu/)  

### Step 1: Set up Docker's apt repository  
```bash
sudo apt-get update  
sudo apt-get install -y ca-certificates curl  
sudo install -m 0755 -d /etc/apt/keyrings  
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc  
sudo chmod a+r /etc/apt/keyrings/docker.asc  
```

### Step 2: Add Docker's repository to apt sources  
```bash
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null  
sudo apt-get update  
```

### Step 3: Install Docker  
```bash
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin  
```

### Step 4: Add the current user to the Docker group  
```bash
sudo usermod -aG docker $USER  
newgrp docker  # Apply changes in the current shell  
```

---

## Installing the NVIDIA Container Toolkit  
Reference: [Official NVIDIA Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)  

### Step 1: Configure the production repository  
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg  
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list  
sudo apt-get update  
```

### Step 2: Install the NVIDIA Container Toolkit  
```bash
sudo apt-get install -y nvidia-container-toolkit  
```

### Step 3: Configure Docker runtime  
```bash
sudo nvidia-ctk runtime configure --runtime=docker  
```

---

## Testing Docker and NVIDIA Container Runtime  
Reference: [Running a Sample Workload](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html#running-a-sample-workload)  

### Step 1: Run a sample CUDA container  
```bash
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi  
```

### Step 2: Expected Output  
Your output should resemble the following:  
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.86.10    Driver Version: 535.86.10    CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            On   | 00000000:00:1E.0 Off |                    0 |
| N/A   34C    P8     9W /  70W |      0MiB / 15109MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
