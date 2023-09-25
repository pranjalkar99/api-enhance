# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Expose port 8000 for the FastAPI app
EXPOSE 8000

# Run the command to start the FastAPI app when the container launches
CMD ["uvicorn", "deploy_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]