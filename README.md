# ELECT: Toward Unsupervised Outlier Model Selection (ICDM 2022)

----


**We are working on a way to upload the data files which is too large for GitHub**. 
For now, you could use the [GoogleDrive](https://drive.google.com/drive/folders/1DO0HB1m-fbN1h4gOYS5O7_8QIe1hKX5a?usp=sharing) version for it.

**Development Status**: **As of 09/24/2022, ELECT is under active development and in its alpha stage. Please follow, star, and fork to get the latest update**!

Given an unsupervised outlier detection (OD) task on a new dataset, how can we automatically select a good outlier detection method and its hyperparameter(s) (collectively called a model)? 
ELECT is a novel unsupervised outlier model selection method.

## How to run?

To run the demo in the two testbeds, first install the required libraries by executing
"pip install -r requirements.txt". 

**Required Dependencies**:

* Python 3.6+
* joblib>=0.14.1
* liac-arff
* lightgbm
* numpy>=1.13
* scipy>=0.19.1
* scikit_learn>=0.19.1
* pandas
* psutil
* pyod>=0.9


To run the demo in the wild testbed, execute: 

.. code-block:: bash
   "python elect_wild.py".

Similarly, to run demo in the controlled testbed, execute:

.. code-block:: bash
   "python elect_controlled.py".

**More file description**:
* initialization_converage.py includes the implementation of coverage driven initialization.
* utility.py includes a set of helper functions.
* intermediate_files folder includes some useful intermediate_files for fast replication.
* testbeds and datasets folder includes the raw file of all datasets.


