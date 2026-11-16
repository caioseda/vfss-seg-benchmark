# Ambiente de treino do vfss-seg-benchmark, para rodar sweeps nos servidores DeepLearning (e, se quiser,
# na delaunay). So o AMBIENTE vai na imagem: o codigo e montado no `docker run` (scripts/remote.py), entao
# a imagem so muda quando o lock do ambiente muda -- e a tag carrega o hash dele e deste arquivo.
#
#   python scripts/remote.py build          # constroi na delaunay
#   python scripts/remote.py setup dl05     # leva a MESMA imagem (docker save | docker load) + dados
#
# Construir uma vez e levar a mesma imagem a todos os hosts, em vez de construir em cada um, garante
# versoes identicas entre eles.
#
# CUDA: o `pytorch-cuda=12.4` do lock traz o runtime; o host precisa so do driver NVIDIA
# (>= 550) e do nvidia-container-toolkit.
FROM mambaorg/micromamba:1.5.10

USER root
# git: o Experiment grava o commit (run.json). rsync/procps: diagnostico dentro do container.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git rsync procps \
 && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

# O lock do env `vfss` da delaunay (python scripts/remote.py lock), nao o environment.yml: as MESMAS
# versoes que treinaram tudo ate aqui. Conda por URL explicita (sem solve); pip sem resolver dependencias,
# porque a lista ja e o ambiente inteiro.
COPY --chown=$MAMBA_USER:$MAMBA_USER conda-explicit.txt pip-requirements.txt /tmp/
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN micromamba install -y -n base -f /tmp/conda-explicit.txt \
 && python -m pip install --no-cache-dir --no-deps -r /tmp/pip-requirements.txt \
 && micromamba clean --all --yes

RUN python -c "import torch, pytorch_lightning, segmentation_models_pytorch, mlflow, omegaconf; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# O container roda com o uid/gid de quem o lanca (--user), que nao existe no /etc/passwd da imagem:
# o PATH nao pode depender do entrypoint, e o HOME vem do `docker run` (/cache/home).
ENV PATH=/opt/conda/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
