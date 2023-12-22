# Breast Cancer Prediction Project

## Overview

This project involves the development of a machine learning model to predict breast cancer based on various features. The logistic regression algorithm is employed for its simplicity and interpretability.

## Table of Contents

- [Getting Started](#getting-started)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

To get started with the project, follow these instructions:

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/AdewuyiDaniels/breast-cancer-prediction.git
    ```

2. Navigate to the project directory:

    ```bash
    cd breast-cancer-prediction
    ```

3. Install the required dependencies (see [Dependencies](#dependencies)).

4. Run the Flask web application or use the provided model for predictions (see [Usage](#usage)).

## Dataset

The dataset used for training and testing the model is the [Coimbra Breast Cancer Dataset]([link-to-dataset](https://www.kaggle.com/datasets/tanshihjen/coimbra-breastcancer)). Ensure that the dataset is available and correctly formatted.

## Dependencies

The project relies on the following Python libraries:

- Flask
- pandas
- scikit-learn
- joblib

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## Usage
Flask Web Application (Optional)
If you want to run the Flask web application for interactive predictions, follow these steps:

1. Upload your dataset using the provided form.
2. Train the model using the uploaded dataset.
3. Make predictions using the trained model.
python app.py
