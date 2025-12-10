#!/bin/bash

# export CUDA_HOME=/is/software/nvidia/cuda-11.8
# export LD_LIBRARY_PATH=/is/software/nvidia/cuda-11.8/lib64:$LD_LIBRARY_PATH
# export PATH=$PATH:/is/software/nvidia/cuda-11.8/bin

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


export PYTHONPATH=$PYTHONPATH:$(pwd)


# Activate conda environment
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Conda not found in standard locations!"
fi


path="./data"
path_set="new_data"

PROGECT_DIR=$(pwd)

# Step 1: Hair processing
conda deactivate && conda activate hairstep

SAM_PATH="./checkpoints/img2hairstep/SAM-models/sam_vit_h_4b8939.pth"
cd ./submodules/external/HairStep
python scripts/img2masks.py --root_real_imgs "$path/$path_set" --checkpoint_sam $SAM_PATH
python scripts/img2strand.py --root_real_imgs "$path/$path_set"

echo "Finished direction maps and silhouette estimation."

# Step 2: Orientation maps and confidence maps
cd $PROGECT_DIR && python ./preprocess_dataset/calc_gabor_mask.py --img_path "$path/$path_set/resized_img" \
    --path_to_save "$path/$path_set/orientation_maps/" \
    --path_to_save_conf "$path/$path_set/confidence_maps"
    
echo "Finished gabor maps estimation."

# Step 3: Depth processing
conda deactivate && conda activate ml-depth-pro

cd $PROGECT_DIR && cd ./submodules/external/ml-depth-pro
depth-pro-run -i "$path/$path_set/resized_img/" -o "$path/$path_set/depth_apple_pro"

echo "Finished depth estimation."

# Step 4: data alignment
cd $PROGECT_DIR && python ./preprocess_dataset/calc_alignment.py --img_path "$path/$path_set/resized_img" \
    --hair_path "$path/$path_set/seg" \
    --all_paths_for_processing seg body_img orientation_maps depth_apple_pro strand_map \
    --gt_img_path "/home/vsklyarova/Projects/Im2Haircut_developer/data/aligned_image.png"

echo "Finished data aligning."

# # Step 5: 3D face reconstruction
export PYTHONPATH=$PROGECT_DIR

conda deactivate && conda activate deep3d_pytorch

cd $PROGECT_DIR
python ./preprocess_dataset/deep3dfacereconstruction_annotate_folder.py --root_path "$path/$path_set/resized_img_aligned" --save_postfix "_aligned"
python ./preprocess_dataset/deep3dfacereconstruction_annotate_folder.py --root_path "$path/${path_set}/resized_img"

# # Step 6: Projection matrix calculation
conda deactivate && conda activate hairstep
cd $PROGECT_DIR &&  python calc_proj_matx.py --root_path "$path/$path_set"
cd $PROGECT_DIR &&  python calc_proj_matx.py --root_path "$path/$path_set"  --save_postfix "_aligned"

echo "Pipeline finished."