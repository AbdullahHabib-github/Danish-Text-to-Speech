FROM python:3.12-slim  
# Base image with a slim Python 3.12 environment

WORKDIR /app  
# Working directory for your app within the container

COPY requirements.txt .  
# Copy only the requirements.txt for efficient caching

RUN pip install -r requirements.txt 
# Install dependencies

COPY . .  
# Copy the rest of your Flask application's code

EXPOSE 7860 
# Standard port for Flask development (change if needed)

CMD ["python","app.py"]  
# Command to start your Flask app
