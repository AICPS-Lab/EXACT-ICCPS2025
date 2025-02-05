# Use a base image compatible with both Windows (AMD64) and Mac (ARM64)
FROM --platform=$TARGETARCH pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime

# Set the working directory inside the container
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Set the default command (change if needed)
CMD ["bash"]
