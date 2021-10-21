# Use image with installed anaconda
FROM continuumio/miniconda3

# Set working directory
WORKDIR /usr/app

# Copy project files
COPY . .

ENTRYPOINT ["./keepalive.sh"] 
