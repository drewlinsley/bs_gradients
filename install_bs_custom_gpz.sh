#!/bin/bash

oscar_path="/cifs/data/tserre_lrs/"
gpznode_path="/media/data_cifs_lrs/"
current_path=$gpznode_path

git_path="https://github.com/"
git clone "${git_path}serre-lab/brain-score.git"
cd brain-score
git checkout hackaton
pip install -e .
cd ..
git clone  "${git_path}serre-lab/result_caching.git"
cd result_caching
git checkout hackaton
pip install -e .
cd ..
git clone "${git_path}serre-lab/model-tools.git"
cd model-tools
git checkout hackaton
pip install -e .
cd ..
git clone "${git_path}serre-lab/brainio_contrib.git"
cd brainio_contrib
git checkout hackaton
pip install -e .
cd ..
git clone  "${git_path}serre-lab/brainio_collection.git"
cd brainio_collection
git checkout hackaton
pip install -e .
cd ..
git clone "${git_path}serre-lab/brainio_base.git"
cd brainio_base
git checkout hackaton
pip install -e .
cd ..
git clone "${git_path}serre-lab/candidate_models.git"
cd candidate_models
git checkout hackaton
pip install -e .
cd ..
pip install einsumt
git clone "${git_path}qbilius/models/" tf-models
echo "export PYTHONPATH='$PYTHONPATH:$(pwd)/tf-models/research/slim'" >> ~/.bashrc
echo "export BRAINIO_HOME='${current_path}projects/prj_brainscore/hackaton2021/.brainio'" >> ~/.bashrc
echo "export RESULTCACHING_DISABLE='1'" >> ~/.bashrc  
echo "export CM_HOME='${current_path}projects/prj_brainscore/hackaton2021/.candidate_models'" >> ~/.bashrc
pip install efficientnet
pip install opencv-python
