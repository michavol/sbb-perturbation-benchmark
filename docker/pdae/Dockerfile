FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install CellSimBench
COPY pyproject.toml /tmp/pyproject.toml
COPY cellsimbench /tmp/cellsimbench
RUN pip install -e /tmp/

# Install pDAE runtime dependencies
RUN pip install --no-cache-dir \
    anndata==0.12.0 \
    scanpy==1.11.3 \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    seaborn \
    matplotlib \
    tqdm \
    gdown \
    "git+https://github.com/antoninschrab/mmdagg.git@baca0a989273b0961b81135b4488fb5695a9e7ab"

# Provide package layout expected by pDAE imports:
# PerturbationExtrapolation.pdae.<module>
RUN mkdir -p /app/PerturbationExtrapolation
COPY cellsimbench/models/pdae/pdae /app/PerturbationExtrapolation/pdae
RUN touch /app/PerturbationExtrapolation/__init__.py

# Copy wrapper entrypoints
COPY docker/pdae/run_model.py /app/run_model.py
COPY docker/pdae/pdae_wrapper.py /app/pdae_wrapper.py

WORKDIR /app
ENV PYTHONPATH="/app:${PYTHONPATH}"

ENTRYPOINT ["python", "/app/run_model.py"]
