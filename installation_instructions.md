# Deep Learning class - installation instructions

Download the miniforge3 distribution for your Operating System
(Windows, macOS or Linux):

   https://github.com/conda-forge/miniforge#miniforge3

miniforge is a conda installation that uses the packages from the conda-forge
installation by default. We recommand using this instead of Anaconda/miniconda3
because conda-forge tends to be more up-to-date and support more platforms than
the default channel of Anaconda/miniconda. However both might work for this
class.

Optional: feel free to create a dedicated conda environment for this class
if you don't want to mess with Python dependencies needed for other classes or projects:

    conda create -n dlclass python=3.9
    conda activate dlclass

Install or update the following packages with the conda command:

    conda install -y tensorflow scikit-learn pandas jupyterlab matplotlib-base
    conda install -y h5py pillow scikit-image lxml pip ipykernel

Check that you can import tensorflow with the python from anaconda:

    python -c "import tensorflow as tf; print(tf.__version__)"
    2.6.0

Note that any tensorflow version from 2.0.0 should work for this class.

If you have several installations of Python on your system (virtualenv, conda
environments...), it can be confusing to select the correct Python environment
from the jupyter interface. You can name this environment for instance
"dlclass" and reference it as a Jupyter kernel:

    python -m ipykernel install --user --name dlclass --display-name dlclass

Ideally: create a new jupyter notebook and check that you can import
the numpy, matplotlib, tensorflow  modules.

To take pictures with the webcam we will also need opencv-python:

    python -m pip install opencv-python

If your laptop does not have a webcam or if opencv does not work, don't worry
this is not mandatory.


# Troubleshooting

In a console check the installation location of the conda command in
your PATH:

    conda info

Read the output of that command to verify that your conda command is installed
where you expect it to be. If it's not the case, you might want to change the
order in your PATH variable (for instance in your $HOME/.bashrc or $HOME/.zshrc
file on Linux and macOS) accordingly.

Check that the pip command in your PATH is the one installed by conda:

    pip show pip

and check that it matches:

    python -m pip show pip

In particular, look at the "Location:" line of pip is a subfolder
of the "environment root:" line from "conda info".

If you cannot solve your installations without the help of fellow students,
please feel free to contact instructors (preferably in the slack channel of the
class) with the outputs of the previous command and include the full error
messages along with the name and version of your operating system.
