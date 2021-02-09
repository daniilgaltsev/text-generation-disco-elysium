FROM python:3.7-slim

# Install a library for Torch to work
RUN apt update && apt install -y --no-install-recommends libtinfo5

# Install Python dependencies
COPY requirements.txt ./requirements.txt
RUN set -ex && pip3 install --no-cache-dir -r requirements.txt

# Create the working directory
RUN set -ex && mkdir /repo
WORKDIR /repo

# Copy the relevant directories to the working directory
COPY text_generation ./text_generation
COPY api ./api

# Add paths to packages
ENV PYTHONPATH /repo
ENV PYTHONPATH /repo/api

# Run the web server
EXPOSE 8080
CMD gunicorn --bind 0.0.0.0:8080 --workers 1 wsgi