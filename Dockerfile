
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

WORKDIR /home/---


ENV DEBIAN_FRONTEND=noninteractive


RUN apt-get update && \
    apt-get install -y curl bzip2 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


RUN rm -rf /opt/conda && \
    curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /miniconda.sh && \
    ls -l /miniconda.sh && \
    head -n 10 /miniconda.sh && \
    chmod +x /miniconda.sh && \
    bash -x /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh



ENV PATH="/opt/conda/bin:$PATH"


COPY sfsrec_env.yaml .
RUN conda update -n base -c defaults conda && \
    conda env create -f sfsrec_env.yaml

SHELL ["conda", "run", "-n", "sfsrec", "/bin/bash", "-c"]


COPY . .

