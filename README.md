# Requirements
## Hardware
This is the PC on which the code was developed
- CPU: Intel Core i7-9700 (8 cores)
- RAM: 32GB
- GPU: nVidia Gefore RTX3070Ti - 8GB (GPU is not required, but code runs faster with it).


## Software
- OS: Windows / Linux
- Python: 3.10.13
- Pip: 23.2.1
- PyTorch: 2.0.1

## Methods of installation
1. **Run from scratch**: Setup the environment from scratch and install all dependencies by yourself. This is more common and consumes less disk space. However, it could lead to software incompatiblity.
2. **Run with Docker**: Prebuilt Docker image is available which contains everything for the code to run on. Downside is that not all systems have Docker installed and it could take up large amount of disk space.

# Run from scratch

1. Install Miniconda: https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html
2. Install PyTorch: https://pytorch.org/get-started/locally/ (Remember to choose GPU version if you have one)
3. Install Dependencies: Install all required libraries with (remember to activate your conda environment before using this command)
```console
pip install --no-cache-dir -r requirements.txt
```
4. Run web demo:

**CPU:**
```console
python app_f3d_l5.py --device cpu
```
**GPU:**
```console
python app_f3d_l5.py --device cuda
```

# Run with Docker
This provides everything to run the model without having to install python dependencies.
However, you should have Docker installed on your PC as a requirement.
## Build docker image
You can build a new docker image before running:
```console
docker build --pull --rm -f "Dockerfile" -t footlastdl:latest "." 
```
Alternatively, you can use prebuilt image with:
```console
docker pull nqhoang/footlastdl:latest
```

## Running

### Run as Training
**Requirements:** Make sure you have two folders named **data** (which stores that training / testing data) and folder **log** to store training checkpoints, history, and logs.
```console
docker run -u $(id -u):$(id -g) --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v ./data:/app/data -v ./log:/app/log footlastdl python train_regression.py --model pointnet2_regression_attn --exp_name attn_oct22 --dataset_dir "data/3D_All_Foot/oct17_f3d_l5/" --num_points 3000 --n_epochs 4000 --finetune --batch_size 32 --lr 1e-4
```
<details>
  <summary>Explanation</summary>
  
 Let's break down the Docker command step by step:

**docker run**: This is the command to run a Docker container.

* **`-u $(id -u):$(id -g)`**: This sets the user for the container. $(id -u) and $(id -g) are shell commands that fetch the current user's UID (User ID) and GID (Group ID), respectively. This ensures that the processes * inside the Docker container run with the same user and group as the host system's current user.
* **--gpus all**: This option enables GPU support in the container and allows it to access all available GPUs on the host.
* **--ipc=host**: This shares the host's inter-process communication (IPC) namespace with the container. This is often used for sharing certain resources between the host and the container.
--**ulimit memlock=-1 --ulimit stack=67108864**: These set ulimit values for the container. memlock=-1 allows the container to use an unlimited amount of locked memory, and stack=67108864 sets the maximum stack size * for processes in the container to 64MB.
* **-it**: This makes the container interactive and allocates a pseudo-TTY (terminal).
* **-v ./data:/app/data -v ./log:/app/log**: This mounts host directories into the container. It maps the local ./data directory to /app/data inside the container and ./log to /app/log.
* **footlastdl**: This is the name of the Docker image that is being run.
* **python train_regression.py**: This is the command that is executed inside the container. It runs a Python script named train_regression.py.
* **--model `pointnet2_regression_attn`**: This specifies the model to be used, which is pointnet2_regression_attn.
* **--exp_name `attn_ln_oct_16`**: This sets the experiment name to attn_ln_oct_16, each experiment contains trained weights, and history.
* **--dataset_dir "data/3D_All_Foot/oct13new/"**: This specifies the directory containing the dataset.
* **--num_points 3000**: This sets the number of points to 3000.
* **--n_epochs 4000**: This sets the number of training epochs to 4000.
* **--finetune**: This indicates that the backend of the model will be fine-tuned, rather than frozen.
* **--batch_size 16**: This sets the batch size to 16.

In summary, the Docker command is running a container based on the footlastdl image, configuring it to use GPU support, sharing specific host resources, and executing a Python script (train_regression.py) with various parameters and options for training a regression model on a 3D foot dataset. The model is pointnet2_regression_attn, and the training is set to run for 4000 epochs with a batch size of 16, while fine-tuning on an existing model.

</details>

### Run as Testing
**Requirements:** Same as training
```console
docker run -u $(id -u):$(id -g) -it -v ./data:/app/data -v ./log:/app/log footlastdl python test_regression.py --exp_name attn_ln --dataset_dir "data/3D_All_Foot/oct13new/"
```
### Run as Inference (without ground truths)
**Requirements:** Same as testing
```console
docker run -u $(id -u):$(id -g) -it -v ./data:/app/data -v ./log:/app/log footlastdl python inference.py --exp_name attn_ln --infer_data_csv "data/3D_All_Foot/oct13new/infer.csv" --device cpu --batch_size 4
```

### Run as Web demo (Gradio)
**Requirements:** Make sure you have a named folder **log** which contains training checkpoints, history, and logs.

**CPU:**
```console
docker run -u $(id -u):$(id -g) -it -v ./log:/app/log -v ./data:/app/data -p 7860:7860 footlastdl python app_f3d_l5.py --device cpu
```

**GPU:**
```console
docker run -u $(id -u):$(id -g) -it -v ./log:/app/log -v ./data:/app/data -p 7860:7860 footlastdl python app_f3d_l5.py --device cuda
```