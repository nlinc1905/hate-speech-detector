# Pull base image
FROM python:3.9-slim

# Set work directory in docker container
WORKDIR /

# Install dependencies
COPY requirements.txt /
RUN pip install -r requirements.txt

# Copy project to working dir in container
COPY . /

# Start dashboard and bolt
RUN ["chmod", "+x", "./startup.sh"]
ENTRYPOINT ./startup.sh
