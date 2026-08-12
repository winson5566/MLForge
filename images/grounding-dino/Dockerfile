FROM mlforge/annotation-trainer:0.1.3

RUN pip install --no-cache-dir \
    boto3==1.42.34 \
    transformers==4.57.3 \
    accelerate==1.12.0 \
    safetensors==0.7.0 \
    torchvision==0.22.1 \
    numpy==2.0.2

COPY src/mlforge_platform /app/mlforge_platform
