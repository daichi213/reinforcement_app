FROM jupyter/scipy-notebook
RUN conda update --all && conda install tensorFlow==2.4.1 lightgbm  && \
conda install -c bioconda piper
#  自然言語処理
# conda install -c conda-forge -y wordcloud nltk plotly xgboost mecab copulae arch-py fastparquet
# 強化学習の実験環境
RUN conda install -c conda-forge -y gym
