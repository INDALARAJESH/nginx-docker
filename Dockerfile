# Stage 1: Build stage with full dependencies
FROM python:3.10-alpine as builder

# Install build dependencies
RUN apk add --no-cache \
    build-base \
    libxml2-dev \
    libxslt-dev \
    gcc \
    g++ \
    libffi-dev \
    musl-dev \
    libmagic \
    jpeg-dev \
    zlib-dev \
    libjpeg \
    poppler-utils \
    poppler-data

# Set the working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -U langchain-community

# Stage 2: Runtime stage with minimal dependencies
FROM python:3.10-alpine

# Install runtime dependencies
RUN apk add --no-cache \
    libmagic \
    jpeg-dev \
    zlib-dev \
    libjpeg \
    poppler-utils \
    poppler-data

# Set the working directory
WORKDIR /app

# Copy only necessary files from the build stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY . .

# Expose port 8501 for Streamlit
EXPOSE 8501

# Set environment variables
ENV GOOGLE_API_KEY=${GOOGLE_API_KEY} \
    CHATGPT=${CHATGPT}

# Run Streamlit when the container launches
CMD ["streamlit", "run", "chatbot.py"]
