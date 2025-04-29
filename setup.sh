# Put cuda env in bashrc file
echo "export CUDA_HOME=/opt/conda/pkgs/cuda-toolkit/"  >> ~/.bashrc
echo "export PATH=$CUDA_HOME/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
