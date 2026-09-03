FROM python:3.12-slim

WORKDIR /app

# System dependencies for numerical packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Run the data fetching script to ensure dataset exists
RUN python src/fetch_data.py

# Ensure model and plots are generated
RUN python src/eda.py
RUN python src/train_model.py
RUN python src/evaluate.py
RUN python src/generate_report.py

# Expose Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
