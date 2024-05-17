# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libopencv-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download the model file from Google Drive
RUN pip install gdown
RUN gdown https://drive.google.com/uc?id=1wHQHMeAPQK-_gjIbJRPnt-oXTKdIZgim -O model.h5

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run the script
CMD ["python", "facerec2.py"]
