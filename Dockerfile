FROM python:3.10

RUN pip install -U onnxsim onnx onnxruntime
RUN pip install -U torch torchaudio torchvision -i https://download.pytorch.org/whl/cpu

RUN pip install -U pytest
WORKDIR /app
COPY tests/ tests/

ENTRYPOINT ["onnxsim"]
