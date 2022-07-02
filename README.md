# MOLECULAR SOLUBILITY PREDICTION WEB APP

This repo was mainly created for learning to create web apps using [streamlit](https://streamlit.io/).

This tool predicts the molecular solubility of a compound given its SMILES string.

It also calculates shap values and plots some graphics to explain the models predictions.

It is possible to acess prediction from a Lineal Regression and a XGBoost models.

## Requirements:

Clone the repository:

```bash
git clone https://github.com/jcorreia11/streamlit-solubility.git
```

Install the requirements:

```text
pandas~=1.4.2
streamlit~=1.10.0
shap~=0.41.0
xgboost~=1.6.1
rdkit-pypi~=2022.3.3
streamlit_shap~=1.0.2
```

If you want to experiment with the [solubility_models.ipynb](solubility_models.ipynb) notebook you also need:
```text
scikit-learn~=1.0.1
matplotlib~=3.5.1
```

# Run the tool in your browser:

Run the following command in your terminal in the [solubility_app.py](solubility_app.py) directory:

```bash
streamlit run solubility_app.py 
```

Paste your SMILES strings in the SMILES input box and explore!


# Deploy (TODO)

1. Heroku
2. Streamlit Sharing