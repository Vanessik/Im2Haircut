# Need to use this to activate conda environments
eval "$(conda shell.bash hook)"

# Save parent dir
PROJECT_DIR=$PWD

# 1) Install HairStep environment and download checkpoints following https://github.com/GAP-LAB-CUHK-SZ/HairStep:

cd $PROJECT_DIR && cd ./submodules/external/HairStep

conda env create -f environment.yml
conda activate hairstep

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt

cd external/3DDFA_V2
sh ./build.sh
cd ../../

mkdir -p ./checkpoints/SAM-models/
wget -P ./checkpoints/SAM-models/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

mkdir -p ./checkpoints/recon3D/
gdown https://drive.google.com/uc?id=1-akuukaYYtJDta24AAqVdgUOGte4EmQf -O ./checkpoints/recon3D/recon3D.zip
unzip ./checkpoints/recon3D/recon3D.zip -d ./checkpoints/recon3D/
rm ./checkpoints/recon3D/recon3D.zip


# 2) Install ml-depth-pro following https://github.com/apple/ml-depth-pro;

cd $PROJECT_DIR && cd ./submodules/external/ml-depth-pro
conda create -n ml-depth-pro -y python=3.9
conda activate ml-depth-pro

pip install -e .

source get_pretrained_models.sh 


# # 3) Install Deep3DFaceRecon_pytorch following https://github.com/sicxu/Deep3DFaceRecon_pytorch:
cd $PROJECT_DIR && cd ./submodules/external/Deep3DFaceRecon_pytorch
git checkout ab35e02

conda env create -f environment.yml
source activate deep3d_pytorch

# Install Arcface Pytorch:
git clone https://github.com/deepinsight/insightface.git
cp -r ./insightface/recognition/arcface_torch ./models/

# Install Nvdiffrast library:
git clone -b v0.3.0 https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
pip install .

pip install facenet-pytorch
