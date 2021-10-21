# ML in a Box

This repo houses a script used to turn any Paperspace Ubuntu 20.04 based Linux instance into a fully functional machine learning environment for interactive development. Users are free to modify and run against their own Paperspace instance, be it Linux headless or Linux desktop. The only requirement is a working Nvidia GPU driver pre-installed. A pre-built template exists in the Paperspace eco-system as a Public Template based on a base Ubuntu 20.04 image.

## Who is this for?

We assume a generic advanced data science user who probably wants GPU access, but not any particular specialized subfield of data science such as computer vision or natural language processing. Such users can build upon this base to create their own stack, or we can create other VMs for subfields, similar to what can be done with Gradient containers.

## Software included

| Category         | Software         | Version                | Install Method | Why / Notes |
| -------------    | -------------    | -------------          | -------------  | ------------- |
| GPU              | NVidia Driver    | 470.63.01              | pre-installed  | Enable Nvidia GPUs. Latest version as of VM creation date |
|                  | CUDA             | 11.4                   | pre-installed  | Nvidia A100 GPUs require CUDA 11+ to work, so 10.x is not suitable |
|                  | cuDNN            | 8.2.4.15-1+cuda11.4    | Ubuntu repo    | Nvidia GPU deep learning library |
|                  | CUDA toolkit     | 10.1.243-3             | apt-get        | Needed for `nvcc` command for cuDNN |
| Infra            | Docker Engine CE | 20.10.8, build 3967b7d | Ubuntu repo    | Docker Enginer community edition |
|                  | NVidia Docker    | 2.6.0-1                | Ubuntu repo    | Enable NVidia GPU in Docker containers |
| Python           | Python           | 3.8.10                 | Done in above  | Most widely used programming language for data science. Version 3.8.10 is already installed by one of the above steps, and is compatible with other software and their versions installed here. |
|                  | pip3             | 20.0.2-5ubuntu1.6      | apt-get        | Enable easy installation of 1000s of other data science, etc., packages. Is a version 21.2 but apt-get install for Python 3.8 gives 20.0.2, which is OK. |
|                  | NumPy            | 1.21.2                 | pip3           | Handle arrays, matrices, etc., in Python |
|                  | Pandas           | 1.3.3                  | pip3           | De facto standard for data science data exploration/preparation in Python |
|                  | Matplotlib       | 3.4.3                  | pip3           | Widely used plotting library in Python for data science, e.g., scikit-learn plotting requires it |
|                  | JupyterLab       | 3.1.12                 | pip3           | De facto standard for data science using Jupyter notebooks |
| Machine learning | H2O-3            | 3.34                   | pip3           | Enables in one place a wide range of ML algorithms outside deep learning at considerably higher performance than scikit-learn: gradient boosted trees, random forest, support vector machine, k-means clustering, generalized linear model (includes logistic regression), isolation forest, etc. Plus auto-ml, model ensembling, and other features. |
|                  | Scikit-learn     | 0.24.2                 | pip3           | Widely used ML library for data science, generally for smaller data or models |
|                  | TensorFlow       | 2.5.0                  | pip3           | Most widely used deep learning library, alongside PyTorch. Note 2.5.0 as 2.6 unclear if works with CUDA 11.4, NVidia, etc. |
|                  | NVidia RAPIDS    | 21.08                  | conda          | GPU acceleration for common ML algorithms |
|                  | (PyTorch)        | (1.9.0)                | source         | **Not installed because doesn't support CUDA 11.4 yet.** Most widely used deep learning library, alongside TensorFlow |

### Licenses

| Software      | License                | Source |
| ------------- | -------------          | ------------- |
| CUDA 	        | NVidia EULA		  	       | https://docs.nvidia.com/cuda/eula/index.html |
| cuDNN         | NVidia EULA            | https://docs.nvidia.com/deeplearning/cudnn/sla/index.html |
| Docker Engine | Apache 2.0 	           | https://github.com/moby/moby/blob/master/LICENSE |
| H2O-3     	   | Apache 2.0 	           | https://github.com/h2oai/h2o-3/blob/master/LICENSE |
| JupyterLab    | New BSD      	         | https://github.com/jupyterlab/jupyterlab/blob/master/LICENSE |
| Matplotlib    | PSF-based      		      | https://matplotlib.org/stable/users/license.html |
| Numpy      	  | New BSD                | https://numpy.org/doc/stable/license.html |
| NVidia Docker | Apache 2.0             | https://github.com/NVIDIA/nvidia-docker/blob/master/LICENSE |
| NVidia Driver | NVidia EULA            | https://www.nvidia.com/en-us/drivers/nvidia-license/ |
| NVidia RAPIDS | "Open source"          | https://developer.nvidia.com/rapids |
| Pandas        | New BSD                | https://github.com/pandas-dev/pandas/blob/master/LICENSE |
| Pip3          | MIT                    | https://github.com/pypa/pip/blob/main/LICENSE.txt |
| PyTorch       | New BSD                | https://github.com/pytorch/pytorch/blob/master/LICENSE |
| Python        | PSF                    | https://en.wikipedia.org/wiki/Python_(programming_language) |
| Scikit-learn  | New BSD                | https://github.com/scikit-learn/scikit-learn/blob/main/COPYING |
| TensorFlow    | Apache 2.0             | https://github.com/tensorflow/tensorflow/blob/master/LICENSE |

Information about license types:

Apache 2.0: https://opensource.org/licenses/Apache-2.0  
MIT: https://opensource.org/licenses/MIT  
New BSD: https://opensource.org/licenses/BSD-3-Clause  
PSF = Python Software Foundation: https://en.wikipedia.org/wiki/Python_Software_Foundation_License

Open source software can be used for commercial purposes: https://opensource.org/docs/osd#fields-of-endeavor .

## Software not included

Other software considered but not included.

The potential data science stack is far larger than any one person will use, for example, the Anaconda Python distribution for data science has over 7500 optional packages, so we don't attempt to cover everything here.

Some generic categories of software not included:

 - Non-data-science software
 - Commercial software
 - Software not licensed to be used on an available VM template
 - Software only used in particular specialized data science subfields (although we assume our users probably want a GPU)

| Category           | Software | Why Not |
| -------------      | ------------- | ------------- |
| Apache             | Kafka, Parquet | |
| Classifiers        | libsvm, XGBoost | H2O contains SVM and GBM, save on installs |
| Collections        | ELKI, GNU Octave, Weka, Mahout | |
| Connectors         | S3, Academic Torrents, Google Drive | |
| Dashboarding       | panel, dash, voila, streamlit | |
| Databases          | MySQL, Hive, PostgreSQL, Prometheus, Neo4j, MongoDB, Cassandra, Redis | No particular infra to connect to databases |
| Deep Learning      | Caffe, Caffe2, Theano, Keras, PaddlePaddle, Chainer, Torch, MXNet | PyTorch and TensorFlow are dominant, rest niche |
| Deployment         | Dash, TFServing, R Shiny, Flask | Use Gradient Deployments |
| Distributed.       | Horovod, OpenMPI | Use Gradient distributed |
| Distributions      | Anaconda | Includes over 250 packages with many licenses |
| Feature store      | Feast | |
| IDEs               | PyCharm, Spyder, RStudio | |
| Image proc         | OpenCV, Pillow, scikit-image | |
| Interpretability   | LIME/SHAP, Fairlearn, AI Fairness 360, InterpretML | |
| Languages          | R, SQL, Java, Julia, C++, JavaScript, Python2, Scala | Python is dominant for data science |
| Monitoring         | Grafana | |
| NLP                | HuggingFace, NLTK, GenSim, spaCy | |
| Notebooks          | Jupyter, Zeppelin | JupyterLab includes Jupyter notebook |
| Orchestrators      | Kubernetes | Use Gradient cluster|
| Partners           | fast.ai | Could add if we want partner functionality |
| Pipelines          | AirFlow, MLFlow, Intake, Kubeflow | |
| Python libraries   | SciPy, statsmodels, pymc3, geopandas, Geopy, LIBSVM | Too many to attempt to cover |
| PyTorch extensions | Lightning | |
| R packages         | ggplot, tidyverse | Could add R if customer demand |
| Recommenders       | TFRS, scikit-surprise | |
| Scalable           | Dask, Numba, Spark 1 or 2, Koalas, Hadoop | |
| TensorFlow         | TF 1.15, Datasets, Recommenders, TensorBoard, TensorRT | Could add TensorFlow 1.x if customer demand. Requires separate tensorflow-gpu for GPU support. |
| Viz                | Bokeh, Plotly, Holoviz (Datashader), Seaborn, Google FACETS, Excalidraw, GraphViz, ggplot2, d3.js | |
