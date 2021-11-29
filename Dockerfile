FROM jupyter/tensorflow-notebook
USER root
RUN apt-get update && apt-get install -y vim
RUN conda update --all && conda install lightgbm  && \
conda install -c bioconda piper && conda install -c conda-forge xgboost plotly gym
RUN conda install -c conda-forge -y gym
