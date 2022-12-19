git pull
pip install setuptools==59.5.0
ln -s /workspace/nuscenes ./data/nuscenes
# For V100 CUDA11.0 
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip uninstall -y mmcv-full 
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html