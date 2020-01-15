# Deep Learning class - installation instructions

Download the Anaconda distribution for your Operating System
(Windows, macOS or Linux):

   - https://www.continuum.io/downloads (~500 MB)
   - Choose **Python 3.7**
   - Choose "64-bit installer"

Follow the instructions of the Anaconda page to install anaconda
on your laptop.

Open a console / terminal and update the following packages with conda:

    conda install python=3.7 numpy scikit-learn jupyter ipykernel matplotlib pip
    conda install pandas h5py pillow scikit-image lxml tensorflow keras

Check that you can import tensorflow with the python from anaconda:

    python3 -c "import tensorflow as tf; print(tf.__version__)"
    2.1.0

Note that tensorflow 2.0.0 should also work.

If you have several installations of Python on your system (virtualenv, conda
environments...), it can be confusing to select the correct Python environment
from the jupyter interface. You can name this environment for instance
"dlclass" and reference it as a Jupyter kernel:

    python3 -m ipykernel install --user --name dlclass --display-name dlclass


Ideally: create a new jupyter notebook and check that you can import
the numpy, matplotlib, tensorflow  modules.

To take pictures with the webcam we will also need opencv-python:

    python3 -m pip install opencv-python

If your laptop does not have a webcam or if opencv does not work, don't worry
this is not mandatory.


# Troubleshooting

In a console check the installation location of the conda command in
your PATH:

    conda info

Read the output of that command to verify that your conda command is
associated with Python 3.7.


Check that the pip command in your PATH is the one installed by conda:

    pip show pip

and check that it matches:

    python3 -m pip show pip

In particular, look at the "Location:" line of pip is a subfolder
of the "environment root:" line from "conda info".

If you cannot solve your installations without the help of fellow students,
please feel free to send an email to the instructors with the outputs of the
previous command and include the full error messages along with the name and
version of your operating system.
