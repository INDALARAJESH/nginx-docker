# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Update and upgrade existing packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt .

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install langchain-community
RUN pip install --no-cache-dir -U langchain-community

# Copy the rest of the working directory contents into the container at /app
COPY . .

# Expose port 8501 for Streamlit
EXPOSE 8501

# Set environment variables
ENV GOOGLE_API_KEY=${GOOGLE_API_KEY}
ENV CHATGPT=${CHATGPT}

# Run Streamlit when the container launches
CMD ["streamlit", "run", "chatbot.py"]
