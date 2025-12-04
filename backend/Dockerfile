# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code into the container
# Since this Dockerfile is inside 'backend/', COPY . . puts all python files directly in /app
COPY . .

# Expose the port
EXPOSE 8000

# Command to run the application
# Note: 'main:app' instead of 'backend.main:app' because main.py is now in the root of /app
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
