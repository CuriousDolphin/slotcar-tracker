# Install dependencies 

## Install ffmpeg
```bash
sudo apt-get install ffmpeg
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

# Run the gradio app
```bash
gradio app.py
```