# OKRA: Encoding Data with (sparse, semi-)Orthogonal K-fRAmes

This repository provides code to implement *OKRA*: a secure and efficient kernel learning framework developed for privacy-preserving analysis in medical imaging. OKRA encodes data using (sparse, semi-)Orthogonal K-frames to support effective, scalable learning in sensitive data environments:

Alice, Bob, and Charlie each generate sets of orthonormal \( k \)-frames using shared seeds, creating matrices AΓ, BΓ, and CΓ representing their data. These frames are orthonormal, ensuring that the inner product between frames from different participants equals the identity matrix, preserving data structure while encoding. A shared permutation is applied to enhance privacy further. Each participant encodes their data by applying this permuted transformation, producing encoded matrices \( A' \), \( B' \), and \( C' \), which they send to a central server. This encoding allows the server to accurately compute kernel functions (Linear, Gaussian, Polynomial, and Rational Quadratic) across the distributed data without compromising privacy under the semi-honest setting.

THis is the code for the paper: [Private, Efficient and Scalable Kernel Learning for Medical Image Analysis]([https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10431639](https://arxiv.org/pdf/2410.15840))

## Installation

Clone the repository and install dependencies:

bash
git clone https://github.com/your-repo/kernel_lib.git
cd kernel_lib
pip install -r requirements.txt

Make sure sklearn, numpy, and scipy are installed.

## Functionality

- Random Matrix Generation: Create random matrices for encoding.
- Data Transformation: Transform data with random matrices for secure encoding.
- Custom Kernel Computation: Compute Gram matrices for various kernel methods.

## Client-Server Architecture

The client-server setup is managed with a Bash script (server_client_classification.sh), allowing multiple runs with configurable parameters.

### Usage

To start the server-client architecture, use:

bash
bash server_client_classification.sh <no_of_runs> <base_port>

**Arguments**:
- <no_of_runs>: Specifies the number of independent runs for model training.
- <base_port>: Defines the base port number for starting the server. Each subsequent run will use an incremented port number based on this base.
One can experiment with number of input parties and number of features. There is also another bash script that can be used for scalability experiemnts. 

# Data:
[Alzheimer's disease OASIS 3](https://www.oasis-brains.org/)

[Blood Cell Images](https://github.com/Shenggan/BCCD_Dataset)

Synthetic datasets by Scikit-learn 
