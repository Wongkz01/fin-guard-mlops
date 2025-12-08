# 1. Base Image: Use an official lightweight Python version
# We explicitly use 3.11 because we know it works with PyTorch!
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy dependencies first (Optimization: Docker caches this layer)
COPY requirements.txt .

# 4. Install Dependencies
# --no-cache-dir keeps the image small
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application code
COPY . .

# 6. Expose the port the app runs on
EXPOSE 8000

# 7. Command to run the application
# host 0.0.0.0 is required for Docker to be accessible from outside
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]