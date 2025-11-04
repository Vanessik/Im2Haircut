# Prerequisites:
#
# 1. Install CUDA 11.8
#    Follow intructions on https://developer.nvidia.com/cuda-11-8-0-download-archive
#    Make sure that
#      -   PATH includes <CUDA_DIR>/bin
#      -   LD_LIBRARY_PATH includes <CUDA_DIR>/lib64
#    If needed, restart bash environment

#    The environment was tested only with this CUDA version
#
# export CUDA_HOME=/is/software/nvidia/cuda-11.8
# export LD_LIBRARY_PATH=/is/software/nvidia/cuda-11.8/lib64
# export PATH=$PATH:/is/software/nvidia/cuda-11.8/bin

# Check common paths for cuda
if command -v nvcc &> /dev/null; then
    CUDA_HOME=$(dirname $(dirname $(which nvcc)))
elif [ -d "/is/software/nvidia/cuda-11.8" ]; then
    CUDA_HOME=/is/software/nvidia/cuda-11.8
else
    echo "CUDA not found! Please install or add nvcc to PATH."
    return 1  # works if sourced
fi

export CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "CUDA detected at: $CUDA_HOME"
echo "LD_LIBRARY_PATH detected at: $LD_LIBRARY_PATH"


# Need to use this to activate conda environments
eval "$(conda shell.bash hook)"

# Save parent dir
PROJECT_DIR=$PWD
GAUSSIAN_HAIRCUT_PATH="submodules/external/GaussianHaircut/ext"

cd $PROJECT_DIR/$GAUSSIAN_HAIRCUT_PATH && git clone https://github.com/facebookresearch/pytorch3d
cd $PROJECT_DIR/$GAUSSIAN_HAIRCUT_PATH/pytorch3d && git checkout 2f11ddc5ee7d6bd56f2fb6744a16776fab6536f7
cd $PROJECT_DIR/$GAUSSIAN_HAIRCUT_PATH && git clone https://github.com/camenduru/simple-knn
cd $PROJECT_DIR/$GAUSSIAN_HAIRCUT_PATH/diff_gaussian_rasterization_hair/third_party && git clone https://github.com/g-truc/glm
cd $PROJECT_DIR/$GAUSSIAN_HAIRCUT_PATH/diff_gaussian_rasterization_hair/third_party/glm && git checkout 5c46b9c07008ae65cb81ab79cd677ecc1934b903

# Install environment
cd $PROJECT_DIR && conda env create -f environment.yml
conda activate im2haircut

conda create -y -n matte_anything \
    pytorch=2.0.0 pytorch-cuda=11.8 torchvision tensorboard timm=0.5.4 opencv=4.5.3 \
    mkl=2024.0 setuptools=58.2.0 easydict wget scikit-image gradio=3.46.1 fairscale \
    -c pytorch -c nvidia -c conda-forge # this worked better than the official installation config
    

# Download Im2Haircut files:
gdown https://drive.google.com/uc?id=1xFjXmdqLIIOGKOW_akxcDVIalTbo-bkg
tar -xzvf data.tar.gz

# Download Im2Haircut checkpoints:
gdown https://drive.google.com/uc?id=1uOuJx8kO22IZS3WTOeA5IQMw4cHXyamg
tar -xzvf pretrained_models.tar.gz