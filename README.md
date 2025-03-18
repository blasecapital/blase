# blase
## blase: expand and standardize your machine learning

## What is it?
`blase` (*'blāz'*) is a TensorFlow-GPU framework for training neural networks on a single, local machine while working with arbitrarily large datasets and enforcing documentation. It removes cloud dependency for small to medium-sized projects by chunking data at every stage—cleaning, preprocessing, training, and evaluation—allowing models to handle more data. You are only limited by your machine's capacity to store the full dataset in drive and a chunk of training data and model weights in RAM. The framework gives you custom control over feature and target engineering, model architecture, and business impact modeling functions for iterative development. Reproducibility is built in by storing custom functions, datasets, model architecture, and arguments, keeping models ready for production.

*`blase` builds upon Blase Capital Management, LLC's `balcones_train` library to make general-purpose deep learning model training pipeline functions.*

## Roadmap
### The `why?`
`blase` will support end-to-end local model training across a variety of use cases to address the following problems:

1. **Project Size** - Some datasets are too large to store in memory but are not big enough to justify paying for cloud services.
2. **Reproduction and Standardization** - Model training is complex and has many steps. It is easy to overlook best practices and documentation, evidenced by the [reproducibility crisis](https://reproducible.cs.princeton.edu/). 
3. **Data Privacy and Security** - Some projects need to work offline for compliance purposes.
4. **Iteration Speed** - Unorganized notebooks and cloud latency or interruptions can slow workflows.

Expected users include healthcare researchers, finance professionals, environmental scientist, manufacturers, academics, and hobbyists. 

### Development
Given the range of use cases, the library needs to function with different kinds of data, so developing with sample datasets is necessary for creating practical functions.

Steps to success:

1. Collect and inspect sample datasets.
2. Outline core modules.
    - Extract, Transform, Load
    - Data preparation
    - Train
    - Evaluate
    - Deploy
3. Iterate - make the functions support a general workflow while enabling customization.

## License
This project is licensed under the BSD 3-Clause License.

### Third-Party Dependencies
This project relies on open-source dependencies specified in `requirements.txt`.  
Each dependency is subject to its own license, which can be found in their respective repositories.
