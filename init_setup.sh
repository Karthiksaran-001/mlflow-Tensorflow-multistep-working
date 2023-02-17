conda create --prefix ./env python=3.7 -y
source E:/anaconda/etc/profile.d/conda.sh # use your username instead of sunny
source activate ./env
# pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip3 install torch torchvision torchaudio -y
pip install -r requirements.txt
conda env export > conda.yaml