# 基础镜像选择
FROM --platform=linux/amd64 gcr.io/kaggle-images/python:latest

# 设置工作目录
WORKDIR /workspace

# 切换至国内的软件源
RUN sed -i 's;http://deb.debian.org/;https://mirrors.bfsu.edu.cn/;g' /etc/apt/sources.list.d/debian.sources
RUN pip config set global.index-url https://mirrors.bfsu.edu.cn/pypi/web/simple

# 复制项目文件到工作目录
COPY requirements.txt .

# 安装依赖包
RUN apt-get -y update
RUN pip install --no-cache-dir -r requirements.txt
