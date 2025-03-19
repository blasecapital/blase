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
Given the range of use cases, the library needs to function with different kinds of data, so developing with sample datasets is necessary for creating practical functions. Furthermore, batched data extracting and loading is required to handle datasets larger than memory.

Steps to success:

1. Collect and inspect sample datasets. The following are suggested datasets and use case.
    - Healthcare:
        - Readmission prediction (classification)
        - Disease detection with images (classification)
    - Finance:
        - Sentiment detection with news/filings (classification, language-based analysis)
        - Trade prediction (timeseries, classification)
        - Home price estimation (regression)
    - Environmental Science:
        - Species identification (classification, fine tune a pre-trained model)
        - Weather prediction (regression/multi-class classification, timeseries)
    - Industry:
        - Defect detection (computer vision)
        - Machine failure from vibration data (regression)
    - Generative/Language:
        - Fine-tune/create small language model
        - Image generation
    - Reinforcement Learning:
        - Gaming env

2. Outline core modules.
    - Extract, Transform, Load
    - Data preparation
        - Inspect, clean, synthesize
    - Train
    - Evaluate
    - Deploy
3. Iterate - make the functions support a general workflow while enabling customization.

## Setup
### Recommended steps using Docker:
**Prerequisites**
1. [Install Docker](https://www.docker.com/products/docker-desktop/)
    - Enable WSL2 Backend if using Windows - Follow [this guide](https://learn.microsoft.com/en-us/windows/wsl/install)
2. [Install VS Code](https://code.visualstudio.com/download) or Docker-compatible IDE
3. Install Docker and Python extensions in VS Code

**Run 'balcones_train'**
1. Clone the repository
   ```sh
   cd <your preferred install directory>
   git clone https://github.com/blasecapital/balcones_train.git
   ```
2. Open VS Code and navigate to the 'blase' directory in 'Explorer'
3. Right-click on 'docker-compose.yml' and select 'Compose Up' to build the container
4. Navigate to the Docker extension, right-click on the 'blase_env' under 'Containers,' and select 'Attach Visual Studio Code'
5. Navigate to the /workspace folder in the 'Explorer' tab to view the project directory

## Contribute
This project is pretty much a clean slate so you can make an impact! Discussions and issues will be updated with topics related to design and to-do's. This project will make it easier for practitioners to make better models - particularly in healthcare - so we can provide the tools to improve lives!

## License
This project is licensed under the BSD 3-Clause License.

### Third-Party Dependencies
This project relies on open-source dependencies specified in `requirements.txt`.  
Each dependency is subject to its own license, which can be found in their respective repositories.
