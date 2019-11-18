FROM ubuntu:16.04
ENV http_proxy="http://devnet-proxy.oa.com:8080"  \
    https_proxy="http://devnet-proxy.oa.com:8080" \
    ftp_proxy="http://devnet-proxy.oa.com:8080"

# System packages
RUN apt-get update && \
    apt-get install -y curl git wget bzip2 libgoogle-perftools-dev graphviz build-essential g++

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /root/miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/root/miniconda/bin:${PATH}
RUN conda update -y conda && conda install mklml
ENV CONDA_PREFIX=/root/miniconda

# Python packages from conda
RUN conda install -y mklml

# install golang
RUN wget --no-check-certificate -qO- https://storage.googleapis.com/golang/go1.8.1.linux-amd64.tar.gz | \
    tar -xz -C /usr/local && \
    mkdir /root/gopath && \
    mkdir /root/gopath/bin && \
    mkdir /root/gopath/src
ENV GOROOT=/usr/local/go GOPATH=/root/gopath
ENV PATH=${GOROOT}/bin:${GOPATH}/bin:${PATH}

RUN go get -u github.com/google/pprof

# install cmake-3.14
RUN echo "wget --no-check-certificate -qO- https://cmake.org/files/v3.14/cmake-3.14.0.tar.gz | \
    tar -xz -C /tmp/ && \
    cd /tmp/cmake-3.14.0 && \
    bash ./bootstrap && make -j$(nproc) && make install && \
    rm -rf /tmp/cmake-3.14.0" > /root/build_cmake.sh && bash /root/build_cmake.sh