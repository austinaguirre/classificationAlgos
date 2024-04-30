# Use an official Miniconda image
FROM continuumio/miniconda3

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Flask using Conda
RUN conda install flask

# Copy the environment file
COPY environment.yml /app/environment.yml

# Create the Conda environment
RUN conda env create -f environment.yml

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Activate the Conda environment and run app.py when the container launches
CMD ["conda", "run", "-n", "mlforfun", "python", "app.py"]
