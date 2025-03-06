FROM python:3.12-slim AS python

# Install dependencies
RUN pip install uv

# Copy the source code
WORKDIR /app

# Run the application
CMD ["uv", "run", "pipeline.py"]