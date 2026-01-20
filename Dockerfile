FROM python:3.11.2-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ultralytics depends on `opencv-python` which pulls in GUI libs (e.g. libGL).
# For containers we want headless OpenCV to avoid `ImportError: libGL.so.1`.
RUN pip uninstall -y opencv-python opencv-contrib-python || true \
	&& pip install --no-cache-dir --force-reinstall opencv-python-headless

COPY . .
