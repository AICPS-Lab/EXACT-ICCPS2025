# Use PyTorch 2.4.1 with CUDA 12.1 and cuDNN 9 (pre-installed)
FROM --platform=$TARGETARCH pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# Set the working directory
WORKDIR /app

# Copy project files
COPY . /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install remaining dependencies (excluding PyTorch)
RUN pip install --no-cache-dir -r requirements.txt

# Convert line endings inside the container
RUN apt update && apt install -y dos2unix
RUN find /app -type f -name "*.sh" -exec dos2unix {} +
# Set the default command
CMD ["bash"]
