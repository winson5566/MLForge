FROM mlforge/annotation-trainer:0.1.3

# Keep the CPU-only torch==2.7.1 inherited from annotation-trainer. Letting
# pip resolve Ultralytics would replace it with a multi-gigabyte CUDA build.
RUN pip install --no-cache-dir --no-deps ultralytics==8.3.0 \
    && pip install --no-cache-dir \
      boto3==1.42.34 \
      numpy==2.0.2 \
      opencv-python-headless==4.11.0.86 \
      torchvision==0.22.1 \
      tqdm==4.70.0 \
      psutil==7.2.2 \
      py-cpuinfo==9.0.0 \
      seaborn==0.13.2 \
      ultralytics-thop==2.1.6

COPY src/mlforge_platform /app/mlforge_platform
