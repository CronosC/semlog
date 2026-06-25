sudo apt-get update && sudo apt-get install -y \
    wget \
    git \
    cmake \
    build-essential \
    bison \
    flex \
    libboost-dev \
    graphviz-dev
    
cd /build/ && \
    wget https://github.com/whitemech/cudd/releases/download/v3.0.0/cudd_3.0.0_linux-amd64.tar.gz && \
    tar -xf cudd_3.0.0_linux-amd64.tar.gz && \
    cd cudd_3.0.0_linux-amd64 && \
    cp -P lib/* /usr/local/lib/ && \
    cp -Pr include/* /usr/local/include/ && \
    cd /build/ && \
    rm -rf cudd_3.0.0_linux-amd64* && \
    wget https://github.com/whitemech/MONA/releases/download/v1.4-19.dev0/mona_1.4-19.dev0_linux-amd64.tar.gz && \
    tar -xf mona_1.4-19.dev0_linux-amd64.tar.gz && \
    cd mona_1.4-19.dev0_linux-amd64 && \
    cp -P lib/* /usr/local/lib/ && \
    cp -Pr include/* /usr/local/include && \
    cd /build/ && \
    rm -rf mona_1.4-19.dev0_linux-amd64* && \
    git clone https://github.com/whitemech/Syft.git && \
    cd Syft && \
    git checkout v0.1.1 && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j$(nproc) && \
    make install && \
    cd /build/ && \
    rm -rf Syft && \
    mkdir /build/lydia && cd /build/lydia && \
    git clone --recursive https://github.com/whitemech/lydia.git /build/lydia && \
    git checkout main && \
    rm -rf build && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-include cstdint" .. && \
    cmake --build . --target lydia-bin -j$(nproc) && \
    make install && \
    cd /build/ && \
    rm -rf lydia && \
    alias lydia='/usr/local/bin/lydia' && \
    echo "export LD_LIBRARY_PATH=/usr/local/lib" >> /etc/profile
