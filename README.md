# 🚗 Slot Car Tracker 

This toy project leverages fast computer vision object detection models to track slot cars in a video or live stream, offering the following features:

- Detection and tracking of slot cars
- Counting the number of laps
- Recording lap times
- Identifying the best lap time

The models are trained with a custom dataset with carrera-go track and cars **[carrera-go 20062549 up-to-speed](https://www.carrera-toys.com/it/product/20062549-up-to-speed)**. 

The currently supported slot cars are:

- 1/42 (grey) F1 Mercedes 
- 1/42 (red) F1 Ferrari


## 🤖 Models

**Actually models are private and not available for public use**.

Proprietary detection models are provided and trained with the **FocoosAI platform** (COMING SOON 🚀🚀):
[FocoosAI](https://focoos.ai), models are available with differents runtimes:

- onnxruntime-cuda
- onnxruntime-tensorrt-fp32
- onnxruntime-tensorrt-fp16

The base FocoosAI models used in this project are:

- fcs_rtdetr
- fcs_det_small





## 📚 Dataset
The dataset, available on [Roboflow Universe](https://universe.roboflow.com/curiousdolphin-nlvx3/carrera-go), is created by extracting frames from slot car race videos. (it is reccomended to extract frames instead of static images due to high speed of the cars)

Classes in the first dataset version:
- 0: red
- 1: grey
- 2: red-out
- 3: grey-out

Classes in the latest dataset version:
- 0: red
- 1: grey

# 📦 Install dependencies 

## Install ffmpeg and cudnn9
```bash
sudo apt-get install ffmpeg cudnn9-cuda-12
```


## Install uv
```bash
pip install uv
```

## create a virtual environment
```bash
uv venv --python 3.12
source .venv/bin/activate
```

## Install python dependencies
```bash
uv pip install -r requirements.txt
```

# 🚀 Run gradio app
```bash
gradio app.py
```

# 🤝 Contributing

Contributions are welcome! Please feel free to fork repo, submit a pull request or create new datasets and models.

