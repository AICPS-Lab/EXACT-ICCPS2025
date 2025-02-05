# Use an official PyTorch image as base 2.1.0-cuda11.8-cudnn8-runtime
FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

# Set a working directory inside the container
WORKDIR /app

# Copy your project files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Set the default command
CMD ["bash"]


