# ML in a Box

Last updated: Sep 16th 2021

This is to update the Paperspace Core "ML-in-a-Box" template VM image from Ubuntu 18.04 to 20.04.

By recording the software choice and scripts used to set up the new template, this should make it easier to update in future, and keep the VM up to date with an appropriate base of ML software for users. (The previous 18.04 template has no record of what was run to create it.)

This also makes us flexible and open to customer feedback, as the script can be straightforwardly altered to add new tools, or remove existing ones, and rerun.

## Who is this for?

We assume a generic advanced data science user who probably wants GPU access, but not any particular specialized subfield of data science such as computer vision or natural language processing. Such users can build upon this base to create their own stack, or we can create other VMs for subfields, similar to what can be done with Gradient containers.

We assume they have access to machines outside of this VM, so non-data-science software used everyday by many people is not included.

Some particular software choices are also influenced by assumed details about a user, and are mentioned in the two tables below.

## Software included

Currently we plan to install the following data science software. The list is in alphabetical order.

Particular attention is paid to enabling the just-acquired Nvidia Ampere A100 and Axxx GPUs to work, Nvidia being our primary provider of GPUs on the Paperspace cloud at present.

**TODO**: Licenses, exact version numbers

| Software | Version | Why / Notes |
| ------------- | ------------- | ------------- |
| Atom | 1.58 | Text editor |
| Chrome | 93.0 | Web browser |
| CUDA | 11.4 | Nvidia A100 GPUs require CUDA 11+ to work, so 10.x is not suitable |
| cuDNN | 8.2 | Nvidia GPU deep learning library |
| Docker CE | 20.10 | Containers. CE = community edition |
| H2O-3 | 3.34 | Enables in one place a wide range of ML algorithms outside deep learning at considerably higher performance than scikit-learn: gradient boosted trees, random forest, support vector machine, k-means clustering, generalized linear model (includes logistic regression), isolation forest, etc. Plus auto-ml, model ensembling, and other features. |
| JupyterLab | 3.1 | De facto standard for data science using Jupyter notebooks |
| Matplotlib | 3.4 | Widely used plotting library in Python for data science |
| NumPy | 1.21 | Handle arrays, matrices, etc., in Python |
| NVidia-Docker | 2.6 | Enable NVidia containers |
| NVidia Driver | R.. | Enable Nvidia GPUs. Latest version as of VM creation date |
| Nvidia RAPIDS | 21.08 | GPU acceleration for common ML algorithms |
| Pandas | 1.3 | De facto standard for data science data exploration/preparation in Python |
| Pip3 | 21.2 | Enable easy installation of 1000s of other data science, etc., packages |
| PyTorch | 1.9 | Most widely used deep learning library, alongside TensorFlow |
| Python | 3.9 | Most widely used programming language for data science. Version 3.9 is compatible with other software and their versions installed here. |
| Scikit-learn | 0.24 | Widely used ML library for data science, generally for smaller data or models |
| TensorFlow | 2.5 | Most widely used deep learning library, alongside PyTorch |

## Software not included

Other software considered but not included.

The potential data science stack is far larger than any one person will use, for example, the Anaconda Python distribution for data science has over 7500 optional packages, so we don't attempt to cover everything here.

Some generic categories of software not included:

 - Non-data-science software
 - Commercial software
 - Software not licensed to be used on an available VM template
 - Software only used in particular specialized data science subfields (although we assume our users probably want a GPU)

**TODO**: Table

## Script

**TODO**: Add script and how to run it on the VM to install the software and make the VM a template

## References

Some useful references in deriving the software stack were:

 - prev stack
 - nvidia stacks working combos
 - medium blog
 - anaconda
 - Nick's own notes/lists
