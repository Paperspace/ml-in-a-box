# ML in a Box

This repo houses a script used to turn any Paperspace Ubuntu 22.04 based Linux instance into a fully functional machine learning environment for interactive development. Users are free to modify and run against their own Paperspace instance, be it Linux headless or Linux desktop. The only requirement is a working Nvidia GPU driver pre-installed. A pre-built template exists in the Paperspace eco-system as a Public Template based on a base Ubuntu 22.04 image.

## Who is this for?

We assume a generic advanced data science user who probably wants GPU access, but not any particular specialized subfield of data science such as computer vision or natural language processing. Such users can build upon this base to create their own stack, or we can create other VMs for subfields, similar to what can be done with Gradient containers.

## Software included

| Category         | Software         | Version                | Install Method | Why / Notes |
| -------------    | -------------    | -------------          | -------------  | ------------- |
| System              | Nvidia Driver    | 535.129.03              | pre-installed | Enable Nvidia GPUs. Latest version as of VM creation date |
|                  | CUDA             | 12.1.1                 | Apt  			| Nvidia A100 GPUs require CUDA 11+ to work, so 10.x is not suitable |
|                  | CUDA toolkit     | 12.1.1                 | Apt            | Needed to work with Nvidia driver |
|                  | cuDNN            | 8.9.3.*-1+cuda12.1     | Apt   			| Additional libarary to enhance CUDA functionality |
|           | Python           | 3.11.6                 | Apt            | Most widely used programming language for data science |
|                  | pip3             | 23.3.1                 | Apt            | Enable easy installation of 1000s of other data science, etc., packages. |
| ML Frameworks    | PyTorch          | 2.1.1                  | pip3           | Most widely used deep learning framework |
|                  | Torchvision      | 0.16.1                 | pip3           | Vision libary for PyTorch |
|                  | Torchaudio       | 2.1.1                  | pip3           | Audio libary for PyTorch |
|                  | TensorFlow       | 2.15.0                 | pip3           | Popular deep learning framework |
| Hugging Face     | Transformers     | 4.35.2                 | pip3           | Popular deep learning library for NLP brought to you by Hugging Face |
|                  | Datasets         | 2.14.5                  | pip3           | A supporting Hugging Face library for datasets and data handling |
|                  | Peft             | 0.6.2                  | pip3           | A Hugging Face Parameter-Efficient Fine-Tuning (PEFT) enables efficient adaptation of pre-trained language models to various downstream applications without fine-tuning all the model's parameters |
|                  | Tokenizers       | 0.13.3                  | pip3           | A Hugging Face library supporting implementations of tokenizers |
|                  | Accelerate       | 0.24.1                  | pip3           | A Hugging Face library used to support model training by abstracting boilerplate code |
|                  | Diffusers        | 0.21.4                  | pip3           | A Hugging Face library used for implementation of diffusion models |
|                  | Safetensors      | 0.4.0                   | pip3           | A Hugging Face library to store tensors safely |
| Supporting Libraries | JupyterLab   | 3.6.5                   | pip3           | De facto standard for data science using Jupyter notebooks |
|                  | BitsandBytes          | 0.41.2                  | pip3           | A lightweight wrapper around CUDA custom functions, in particular 8-bit optimizers |
|                  | Cloudpickle      | 2.2.1                  | pip3           | Makes it possible to serialize Python constructs not supported by the default pickle module |
|                  | Scikit-image     | 0.21.0                 | pip3           | Collection of algorithms for image processing |
|                  | Scikit-learn     | 1.3.0                  | pip3           | Widely used ML library for data science, generally for smaller data or models |
|                  | Matplotlib       | 3.7.3                  | pip3           | Widely used plotting library in Python for data science, e.g., scikit-learn plotting requires it |
|                  | IPywidgets       | 8.1.1                  | pip3           | Interactive HTML widgets for Jupyter notebooks and the IPython kernel | 
|                  | Cython           | 3.0.2                  | pip3           | Enables writing C extensions for Python |  
|                  | tqdm             | 4.66.1                 | pip3           | Fast, extensible progress meter |  
|                  | gdown            | 4.7.1                  | pip3           | Google drive direct download of big files |  
|                  | XGBoost          | 1.7.6                  | pip3           | An optimized distributed gradient boosting library |
|                  | Pillow           | 9.5.0                 | pip3           | Python imaging library |  
|                  | seaborn          | 0.12.2                 | pip3           | Python visualization library based on matplotlib |
|                  | SQLAlchemy       | 2.0.21                 | pip3           | Python SQL toolkit and Object Relational Mapper that gives application developers the full power and flexibility of SQL |  
|                  | spaCy            | 3.6.1                  | pip3           | library for advanced Natural Language Processing in Python and Cython |  
|                  | nltk             | 3.8.1                  | pip3           | Natural Language Toolkit (NLTK) is a Python package for natural language processing |  
|                  | boto3            | 1.28.51                | pip3           | Amazon Web Services (AWS) Software Development Kit (SDK) for Python |  
|                  | tabulate         | 0.9.0                  | pip3           | Pretty-print tabular data in Python |  
|                  | future           | 0.18.3                 | pip3           | The missing compatibility layer between Python 2 and Python 3 |    
|                  | jsonify          | 0.5                    | pip3           | Provides the ability to take a .csv file as input and outputs a file with the same data in .json format |  
|                  | opencv-python    | 4.8.0.76               | pip3           | Includes several hundreds of computer vision algorithms |  
|                  | pyyaml           | 5.4.1                  | pip3           | YAML parser and emitter for Python |     
|                  | Sentence Transformers | 2.2.2             | pip3           | A ML framework for sentence, paragraph and image embeddings |
|                  | wandb            | 0.15.10                | pip3           | CLI and library to interact with the Weights & Biases API (model tracking) |
|                  | Deepspeed        | 0.10.3                  | pip3           | A DL optimization library for PyTorch designed to train large distrubuted models with better parallelism |
|                  | CuPyCUDA12x        | 12.2.0                  | pip3           | A NumPy/SciPy-compatible array library for GPU-accelerated computing with Python |
|                  | timm             | 0.9.7                  | pip3           | Deep-learning library that hosts a collection of SOTA computer vision models and tools |
|                  | OmegaConf        | 2.3.0                  | pip3           | A hierarchical configuration system, with support for merging configurations from multiple sources |
|                  | SciPy            | 1.11.2                 | pip3           | Fundamental algorithms for scientific computing in Python |
|                  | gradient         | 2.0.6                  | pip3           | CLI and Python SDK for Paperspace Core and Gradient |

### Licenses

| Software              | License                | Source |
| ---------------       | -------------          | ------------- |
| Nvidia Driver         | Nvidia EULA            | https://www.nvidia.com/en-us/drivers/nvidia-license/ |
| Python                | PSF                    | https://en.wikipedia.org/wiki/Python_(programming_language) |
| Pip3                  | MIT                    | https://github.com/pypa/pip/blob/main/LICENSE.txt |
| CUDA 	                | NVidia EULA		  	 | https://docs.nvidia.com/cuda/eula/index.html |
| cuDNN                 | NVidia EULA            | https://docs.nvidia.com/deeplearning/cudnn/sla/index.html |
| PyTorch               | New BSD                | https://github.com/pytorch/pytorch/blob/master/LICENSE |
| TensorFlow            | Apache 2.0             | https://github.com/tensorflow/tensorflow/blob/master/LICENSE |
| Transformers          | Apache 2.0             | https://github.com/huggingface/transformers/blob/main/LICENSE |
| Datasets              | Apache 2.0             | https://github.com/huggingface/datasets/blob/main/LICENSE |
| Peft                  | Apache 2.0             | https://github.com/huggingface/peft/blob/main/LICENSE |
| Tokenizers            | Apache 2.0             | https://github.com/huggingface/tokenizers/blob/main/LICENSE |
| Accelerate            | Apache 2.0             | https://github.com/huggingface/accelerate/blob/main/LICENSE |
| Diffusers             | Apache 2.0             | https://github.com/huggingface/diffusers/blob/main/LICENSE |
| Safetensors           | Apache 2.0             | https://github.com/huggingface/safetensors/blob/main/LICENSE |
| JupyterLab            | New BSD      	         | https://github.com/jupyterlab/jupyterlab/blob/master/LICENSE |
| BitsandBytes          | MIT                    | https://github.com/TimDettmers/bitsandbytes/blob/main/LICENSE |
| Cloudpickle           | New BSD                | https://github.com/cloudpipe/cloudpickle/blob/master/LICENSE |
| Scikit-image          | New BSD                | https://github.com/scikit-image/scikit-image/blob/main/LICENSE.txt |
| Scikit-learn          | New BSD                | https://github.com/scikit-learn/scikit-learn/blob/main/COPYING |
| Matplotlib            | PSF-based      		 | https://matplotlib.org/stable/users/license.html |
| IPywidgets            | New BSD                | https://github.com/jupyter-widgets/ipywidgets/blob/master/LICENSE |
| Cython                | Apache 2.0             | https://github.com/cython/cython/blob/master/LICENSE.txt |
| tqdm                  | MIT                    | https://github.com/tqdm/tqdm/blob/master/LICENCE |
| gdown                 | MIT                    | https://github.com/wkentaro/gdown/blob/main/LICENSE |
| XGBoost               | Apache 2.0             | https://github.com/dmlc/xgboost/blob/master/LICENSE |
| Pillow                | HPND                   | https://github.com/python-pillow/Pillow/blob/main/LICENSE |
| seaborn               | New BSD                | https://github.com/mwaskom/seaborn/blob/master/LICENSE.md |
| SQLAlchemy            | MIT                    | https://github.com/sqlalchemy/sqlalchemy/blob/main/LICENSE |
| spaCy                 | MIT                    | https://github.com/explosion/spaCy/blob/master/LICENSE |
| nltk                  | Apache 2.0             | https://github.com/nltk/nltk/blob/develop/LICENSE.txt |
| boto3                 | Apache 2.0             | https://github.com/boto/boto3/blob/develop/LICENSE |
| tabulate              | MIT                    | https://github.com/astanin/python-tabulate/blob/master/LICENSE |
| future                | MIT                    | https://github.com/PythonCharmers/python-future/blob/master/LICENSE.txt |
| jsonify               | MIT                    | https://pypi.org/project/jsonify/0.5/#data |
| opencv-python         | MIT                    | https://github.com/opencv/opencv-python/blob/4.x/LICENSE.txt |
| pyyaml                | MIT                    | https://github.com/yaml/pyyaml/blob/master/LICENSE |
| Sentence Transformers | Apache 2.0             | https://github.com/UKPLab/sentence-transformers/blob/master/LICENSE |
| wandb                 | MIT                    | https://github.com/wandb/wandb/blob/master/LICENSE |
| Deepspeed             | Apache 2.0             | https://github.com/microsoft/DeepSpeed/blob/master/LICENSE |
| CuPyCUDA12x           | MIT                    | https://github.com/cupy/cupy/blob/main/LICENSE |
| Timm                  | Apache 2.0	         | https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE |
| OmegaConf             | New BSD                | https://github.com/omry/omegaconf/blob/master/LICENSE |
| SciPy                 | New BSD                | https://github.com/scipy/scipy/blob/main/LICENSE.txt |
| gradient              | ISC                    | https://github.com/Paperspace/gradient-cli/blob/master/LICENSE.txt |


Information about license types:

Apache 2.0: https://opensource.org/licenses/Apache-2.0  
MIT: https://opensource.org/licenses/MIT  
New BSD: https://opensource.org/licenses/BSD-3-Clause  
PSF = Python Software Foundation: https://en.wikipedia.org/wiki/Python_Software_Foundation_License
HPND = Historical Permission Notice and Disclaimer: https://opensource.org/licenses/HPND
ISC: https://opensource.org/licenses/ISC

Open source software can be used for commercial purposes: https://opensource.org/docs/osd#fields-of-endeavor.


## Software not included

Other software considered but not included.

The potential data science stack is far larger than any one person will use so we don't attempt to cover everything here.

Some generic categories of software not included:

 - Non-data-science software
 - Commercial software
 - Software not licensed to be used on an available VM template
 - Software only used in particular specialized data science subfields (although we assume our users probably want a GPU)

| Category           | Software | Why Not |
| -------------      | ------------- | ------------- |
| Apache             | Kafka, Parquet | |
| Classifiers        | libsvm | H2O contains SVM and GBM, save on installs |
| Collections        | ELKI, GNU Octave, Weka, Mahout | |
| Connectors         | Academic Torrents | |
| Dashboarding       | panel, dash, voila, streamlit | |
| Databases          | MySQL, Hive, PostgreSQL, Prometheus, Neo4j, MongoDB, Cassandra, Redis | No particular infra to connect to databases |
| Deep Learning      | Caffe, Caffe2, Theano, PaddlePaddle, Chainer, MXNet | PyTorch and TensorFlow are dominant, rest niche |
| Deployment         | Dash, TFServing, R Shiny, Flask | Use Gradient Deployments |
| Distributed.       | Horovod, OpenMPI | Use Gradient distributed |
| Feature store      | Feast | |
| IDEs               | PyCharm, Spyder, RStudio | |
| Interpretability   | LIME/SHAP, Fairlearn, AI Fairness 360, InterpretML | |
| Languages          | R, SQL, Julia, C++, JavaScript, Python2, Scala | Python is dominant for data science |
| Monitoring         | Grafana | |
| NLP                | GenSim | |
| Notebooks          | Jupyter, Zeppelin | JupyterLab includes Jupyter notebook |
| Orchestrators      | Kubernetes | Use Gradient cluster|
| Partners           | fast.ai | Could add if we want partner functionality |
| Pipelines          | AirFlow, MLFlow, Intake, Kubeflow | |
| Python libraries   | statsmodels, pymc3, geopandas, Geopy, LIBSVM | Too many to attempt to cover |
| PyTorch extensions | Lightning | |
| R packages         | ggplot, tidyverse | |
| Recommenders       | TFRS, scikit-surprise | |
| Scalable           | Dask, Numba, Spark 1 or 2, Koalas, Hadoop | |
| TensorFlow         | TF 1.15, Recommenders, TensorBoard, TensorRT | |
| Viz                | Bokeh, Plotly, Holoviz (Datashader), Google FACETS, Excalidraw, GraphViz, ggplot2, d3.js | |
