# ML in a Box

**IN PROGRESS: Repo is not finished**

Last updated: Sep 18th 2021

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

**TODO**: Licenses, exact version numbers when installed

| Software | Version | Why / Notes |
| ------------- | ------------- | ------------- |
| NVidia Driver | 470.63.01 | Enable Nvidia GPUs. Latest version as of VM creation date |
| CUDA | 11.4 | Nvidia A100 GPUs require CUDA 11+ to work, so 10.x is not suitable |
| cuDNN | 8.2.4.15-1+cuda11.4 | Nvidia GPU deep learning library. Also CUDA toolkit. |
| Docker Engine CE | 20.10.8, build 3967b7d | Docker Enginer community edition |
| NVidia-Docker | 2.6.0 | Enable NVidia GPU in Docker containers |
| Python | 3.8.10 | Most widely used programming language for data science. Version 3.8.10 is already installed by one of the above steps, and is compatible with other software and their versions installed here. |
| pip3 | 20.0.2 | Enable easy installation of 1000s of other data science, etc., packages. Is a version 21.2 but apt-get install for Python 3.8 gives 20.0.2, which is OK. |
| NumPy | 1.21.2 | Handle arrays, matrices, etc., in Python |
| Pandas | 1.3.3 | De facto standard for data science data exploration/preparation in Python |
| Matplotlib | 3.4.3 | Widely used plotting library in Python for data science, e.g., scikit-learn plotting requires it |
| JupyterLab | 3.1.12 | De facto standard for data science using Jupyter notebooks |
| H2O-3 | 3.34 | Enables in one place a wide range of ML algorithms outside deep learning at considerably higher performance than scikit-learn: gradient boosted trees, random forest, support vector machine, k-means clustering, generalized linear model (includes logistic regression), isolation forest, etc. Plus auto-ml, model ensembling, and other features. |
| Scikit-learn | 0.24.2 | Widely used ML library for data science, generally for smaller data or models |
| **TODO** | | |
| TensorFlow | 2.5.0 | Most widely used deep learning library, alongside PyTorch |
| Nvidia RAPIDS | 21.08 | GPU acceleration for common ML algorithms |
| PyTorch | 1.9 | Most widely used deep learning library, alongside TensorFlow |
| Atom | 1.58 | Text editor. Has built-in support for Git. |
| Chrome | 93.0 | Web browser, e.g., for JupyterLab |

## Software not included

Other software considered but not included.

The potential data science stack is far larger than any one person will use, for example, the Anaconda Python distribution for data science has over 7500 optional packages, so we don't attempt to cover everything here.

Some generic categories of software not included:

 - Non-data-science software
 - Commercial software
 - Software not licensed to be used on an available VM template
 - Software only used in particular specialized data science subfields (although we assume our users probably want a GPU)

| Category | Software | Why Not |
| ------------- | ------------- | ------------- |
| Apache | Kafka, Parquet | |
| Classifiers | libsvm, XGBoost | H2O contains SVM and GBM, save on installs |
| Collections | ELKI, GNU Octave, Weka, Mahout | |
| Connectors | S3, Academic Torrents, Google Drive | |
| Dashboarding | panel, dash, voila, streamlit | |
| Databases | MySQL, Hive, PostgreSQL, Prometheus, Neo4j, MongoDB, Cassandra, Redis | No particular infra to connect to databases |
| Deep Learning | Caffe, Caffe2, Theano, Keras, PaddlePaddle, Chainer, Torch, MXNet | PyTorch and TensorFlow are dominant, rest niche |
| Deployment | Dash, TFServing, R Shiny, Flask | Use Gradient Deployments |
| Distributed | Horovod, OpenMPI | Use Gradient distributed |
| Distributions | Anaconda | Includes over 250 packages with many licenses |
| Feature store | Feast | |
| IDEs | PyCharm, Spyder, RStudio | |
| Image proc | OpenCV, Pillow, scikit-image | |
| Interpretability | LIME/SHAP, Fairlearn, AI Fairness 360, InterpretML | |
| Languages | R, SQL, Java, Julia, C++, JavaScript, Python2, Scala | Python is dominant for data science |
| Monitoring | Grafana | |
| NLP | HuggingFace, NLTK, GenSim, spaCy | |
| Notebooks | Jupyter, Zeppelin | JupyterLab includes Jupyter notebook |
| Orchestrators | Kubernetes | Use Gradient cluster|
| Partners | fast.ai | Could add if we want partner functionality |
| Pipelines | AirFlow, MLFlow, Intake, Kubeflow | |
| Python libraries | SciPy, statsmodels, pymc3, geopandas, Geopy, LIBSVM | Too many to attempt to cover |
| PyTorch extensions | Lightning | |
| R packages | ggplot, tidyverse | Could add R if customer demand |
| Recommenders | TFRS, scikit-surprise | |
| Scalable | Dask, Numba, Spark 1 or 2, Koalas, Hadoop | |
| TensorFlow | TF 1.15, Datasets, Recommenders, TensorBoard, TensorRT | Could add TensorFlow 1.x if customer demand. Requires separate tensorflow-gpu for GPU support. |
| Viz | Bokeh, Plotly, Holoviz (Datashader), Seaborn, Google FACETS, Excalidraw, GraphViz, ggplot2, d3.js | |

## Script

**TODO**: Add script info and how to run it on the VM to install the software and make the VM a template

## References

Some useful references in deriving the software stack were:

 - Our previous ML-in-a-Box's [list of contents](https://support.paperspace.com/hc/en-us/articles/115002305973)
 - Nvidia containers with working combinations of CUDA, cuDNN, etc., with other data science software, e.g., [here](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html), [here](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel_21-08.html#rel_21-08), or [here](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_21-08.html#rel_21-08)
 - The Anaconda Python distribution's [overview](https://www.anaconda.com/open-source) (there is also a list of its data science packages)
 - Medium blog on a [data science stack](https://dev.to/minchulkim87/my-data-science-tech-stack-2020-1poa)
 - Various other online articles, forums, etc.
 - Nick's own notes & lists
