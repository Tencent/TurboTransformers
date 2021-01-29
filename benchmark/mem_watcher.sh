watch -n 0.001 'nvidia-smi --id=0  --query-compute-apps=used_memory --format=csv -lms 1 | tee -a gpu.log'
