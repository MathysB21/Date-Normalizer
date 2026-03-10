# Create Virtual Environment

```bash
python -m venv venv
```

# Activate the Virtual Environment

```bash
.\venv\Scripts\activate
```

# PyTorch

I have a CUDA GPU, install this inside your venv and not torch as this will most likely install a CPU version

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

# Check for PyTorch

```bash
python

import torch

torch.cuda.is_available()
```

Note that you need to see True here

# Install Requirements

```bash
pip install -r requirements.txt
```

# Generate data for training

```bash
python src/generate_data.py
```

# Run training script

```bash
python src/train_t5.py
```

# Test inference

```bash
python src/infer.py
```

# Run Uvicorn API for web interface

```bash
uvicorn src.api:app --reload
```

Go to http://127.0.0.1:8000 and Try out the endpoint. No newlines or unescaped characters are allowed, they will error out
