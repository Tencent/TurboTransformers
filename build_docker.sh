docker build  --build-arg http_proxy="http://devnet-proxy.oa.com:8080" --build-arg  https_proxy="http://devnet-proxy.oa.com:8080" -t fast_transformer/fast_transformer:v0.1 -f Dockerfile .

