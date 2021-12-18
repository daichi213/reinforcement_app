FROM jupyter/tensorflow-notebook
USER root
RUN apt-get update && apt-get install -y vim bat
# batcatコマンドのセットアップ
RUN mkdir -p ~/.local/bin
RUN ln -s /usr/bin/batcat ~/.local/bin/bat
# RUN conda update --all && conda install lightgbm  && \
# conda install -c bioconda piper && conda install -c conda-forge xgboost plotly
RUN conda install -c conda-forge -y gym python-utils
