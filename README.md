# Titanic-Survival-Prediction

This project is a machine learning-based survival predictor for Titanic passengers, using scikit-learn and pandas for data processing and model training on the classic Titanic dataset.

## Project Overview

The main objective of this project is to predict whether a passenger on the Titanic would survive or not, based on their personal details and ticket information. The project covers the complete pipeline: loading the dataset, cleaning it, exploring important patterns, training a robust classifier, and analyzing the results through meaningful visualizations.

## Features

- Loads and preprocesses the Titanic dataset (from open public source)
- Handles missing values and encodes categorical features for modeling
- Trains a Random Forest Classifier to predict survival
- Evaluates the trained model with validation metrics and confusion matrix
- Supports sample passenger predictions with probability outputs
- Visualizes both data distribution and model performance for better understanding

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- time (standard library)
- ipython (for pretty output in notebooks)

Install all dependencies using:

```
pip install -r requirements.txt
```

## How to Run

1. **Clone the repository:**
    ```
    git clone https://github.com/VarunGupta86/Titanic-Survival-Prediction-Using-ML.git
    cd Titanic-Survival-Prediction-Using-ML
    ```

2. **Install the dependencies:**
    ```
    pip install -r requirements.txt
    ```

3. **Run the notebook or script:**
    - Open `Titanic Survival Prediction Using ML.ipynb` in Jupyter/Colab or click the link to view it in colab:[Click Here 👆](https://colab.research.google.com/drive/1ar74puYpggChUDWTqiyVOqr-BMaX4ACR?usp=sharing)
    - If using the `.py` script, run: `main.py` in your IDE.

---

## 📝 Usage

- The notebook/script walks you through:
    - Loading and preprocessing Titanic passenger data
    - Exploratory data analysis with visualizations
    - Feature encoding & cleanup
    - Model training & validation
    - Visualizing results: feature importance, confusion matrix, survival breakdowns
    - Running predictions for random validation samples

---

## Output & Results

- Achieved ~80% classification accuracy on the validation set
- Includes sample predictions showing probability of survival for random passengers
- Features plenty of visualizations showing key trends and model insights (see `sample_visualizations/` for saved images)

## Presentation

See `Titanic-Survival-Prediction-Using-ML(Presentation).pdf` for slides summarizing the project.

---
