import pickle

import pandas as pd
from rdkit.Chem import MolFromSmiles, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

import streamlit as st

from streamlit_shap import st_shap
import shap
from xgboost import XGBRegressor


def calculate_descriptors(smiles):
    """
    Calculates the descriptors for a given molecule.
    """
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    generated_features = []
    for sm in smiles:
        mol = MolFromSmiles(sm)
        if mol:
            generated_features.append(calc.CalcDescriptors(mol))
        else:
            generated_features.append(None)
    return pd.DataFrame(generated_features, columns=calc.GetDescriptorNames(), index=smiles)


def load_original_dataset():
    """
    Loads the original dataset.
    """
    df = pd.read_csv('data/processed_logs.csv', header=0, sep=',')
    X = df.drop(['smiles', 'logS'], axis=1)
    Y = df['logS']
    return X, Y


def remove_invalid(smiles):
    """
    Removes invalid molecules from the dataset.
    """
    valid = [sm for sm in smiles if MolFromSmiles(sm)]
    if len(valid) == len(smiles):
        return smiles, "All provided SMILES are valid!"
    return valid, "Some SMILES are invalid! Showing results for valid SMILES only!"


st.write("""
# MOLECULAR SOLUBILITY PREDICTION WEB APP
This app predicts the **Solubility (LogS)** values of molecules!
***
""")

######################
# SIDE PANEL
######################

st.sidebar.header('USER INPUTS:')

# Read SMILES input
SMILES_input = "CC(=O)OC1=CC=CC=C1C(=O)O\nC1=CC=C(C=C1)C=O"

SMILES = st.sidebar.text_area("SMILES input:", SMILES_input)
SMILES = SMILES.split('\n')
SMILES, msg = remove_invalid(SMILES)
st.sidebar.write(msg)

######################
# MAIN PANEL
######################

st.header('Input SMILES')
SMILES

model = st.sidebar.radio(
    "Which model do you want to use?",
    ('Linear Regression', 'XGBoost'))


# Calculate molecular descriptors
st.header('Computed molecular descriptors')
X = calculate_descriptors(SMILES)
X

orig_X, orig_Y = load_original_dataset()

if model == 'Linear Regression':
    ######################
    # Pre-built model
    ######################

    # Reads saved model
    trained_model = pickle.load(open('models/solubility_model.pkl', 'rb'))

    # Apply model to make predictions
    predictions = trained_model.predict(X)

    st.header('Predicted LogS values')
    preds = pd.DataFrame(predictions, columns=['Predicted LogS'], index=SMILES)
    preds


    # compute SHAP values
    st.header('Shap values')

    st.write("Individual shap values for predictions: ")
    option = st.selectbox(
        'For which compound do you want to see the shap values?',
        (SMILES))
    # shap predictions
    explainer = shap.Explainer(trained_model, X)
    shap_values = explainer(X)
    idx = SMILES.index(option)
    st_shap(shap.plots.waterfall(shap_values[idx]), height=300)

    st.write("Shap values for the entire train dataset: ")
    # shap all dataset
    all_explainer = shap.Explainer(trained_model, orig_X)
    all_shap_values = explainer(orig_X)
    st_shap(shap.plots.beeswarm(all_shap_values), height=300)

elif model == 'XGBoost':
    ######################
    # Pre-built model
    ######################

    # Reads saved model
    trained_model = XGBRegressor()
    trained_model.load_model("models/xgboost_regressor.json")

    # Apply model to make predictions
    predictions = trained_model.predict(X)

    st.header('Predicted LogS values')
    preds = pd.DataFrame(predictions, columns=['Predicted LogS'], index=SMILES)
    preds

    # compute SHAP values
    st.header('Shap values')

    st.write("Individual shap values for predictions: ")
    option = st.selectbox(
        'For which compound do you want to see the shap values?',
        (SMILES))
    # shap predictions
    explainer = shap.Explainer(trained_model, X)
    shap_values = explainer(X)
    idx = SMILES.index(option)
    st_shap(shap.plots.waterfall(shap_values[idx]), height=300)

    st.write("Shap values for the entire train dataset: ")
    # shap all dataset
    all_explainer = shap.Explainer(trained_model, orig_X)
    all_shap_values = explainer(orig_X)
    st_shap(shap.plots.beeswarm(all_shap_values), height=300)

    explainer = shap.TreeExplainer(trained_model)
    shap_values = explainer.shap_values(orig_X)

    st_shap(shap.force_plot(explainer.expected_value, shap_values[0, :], orig_X.iloc[0, :]), height=200, width=1000)
    st_shap(shap.force_plot(explainer.expected_value, shap_values[:1000, :], orig_X.iloc[:1000, :]), height=400, width=1000)
