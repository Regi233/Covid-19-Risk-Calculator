import json
import base64
import os

import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import dash_table


# For plotting risk indicator and for creating waterfall plot
import plotly.graph_objs as go
#
import plotly.express as px
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from R_Net import R_Net

# To import pkl file model objects
import joblib
import pickle

from layout_helper import run_standalone_app

text_style = {
    'color': "#506784",
    'font-family': 'Open Sans'
}

# Load model
current_folder = os.path.dirname(__file__)
protein_clf = joblib.load(os.path.join(current_folder, './assets/clf_20N_Protein.joblib'))
meta_clf = joblib.load(os.path.join(current_folder, './assets/clf_20N_meta.joblib'))
mlp_model = R_Net(20, 20).to("cpu")
mlp_model.load_state_dict(torch.load("./assets/mlp_model.pth"))
mlp_model.eval()
mlp_dataset_obj = torch.load("./assets/torch_dataset.data")
clinical_clf = joblib.load(os.path.join(current_folder, './assets/clf_clinical.joblib'))
# load the scaler
scaler_protein = pickle.load(open('./assets/scaler_protein.pkl', 'rb'))
scaler_meta = pickle.load(open('./assets/scaler_meta.pkl', 'rb'))

DATAPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')





df = pd.read_csv('./data/severity.csv')
df['id'] = df.index
Su_Protein_Data = pd.read_csv(os.path.join(DATAPATH, 'Su_Protein_Data.csv'))
Filbin_Protein_Data = pd.read_csv(os.path.join(DATAPATH, 'Filbin_Protein_Data.csv'))
Su_Metabolic_Data= pd.read_csv(os.path.join(DATAPATH, 'Su_Metabolic_Data.csv'))
Shen_Metabolic_Data = pd.read_csv(os.path.join(DATAPATH, 'Shen_Metabolic_Data.csv'))

p_options = []
for col in Su_Protein_Data.columns[1:]:
    # p_options.append({'label':'{}'.format(col, col), 'value':col})
    p_options.append(col)

m_options = []
for col in Su_Metabolic_Data.columns[1:]:
    m_options.append(col)

fnameDict = {"Su_Protein_Data":p_options, "Filbin_Protein_Data":p_options, "Su_Metabolic_Data": m_options, 'Shen_Metabolic_Data': m_options}

names = list(fnameDict.keys())
nestedOptions = fnameDict[names[0]]



DATASETS = {
    'Su_Protein_Data': Su_Protein_Data,
    'Filbin_Protein_Data': Filbin_Protein_Data,
    'Su_Metabolic_Data': Su_Metabolic_Data,
    'Shen_Metabolic_Data': Shen_Metabolic_Data
 
}






def description():
    return 'View multiple sequence alignments of genomic or protenomic sequences.'


def header_colors():
    return {
        'bg_color': '#0C4142',
        'font_color': 'white',
    }


def layout():
    return html.Div(id='alignment-body', className='app-body', children=[
        html.Div([
            html.Div(id='alignment-control-tabs', className='control-tabs', children=[
                dcc.Tabs(
                    id='alignment-tabs', value='what-is',
                    children=[
                        dcc.Tab(
                            label='About',
                            value='what-is',
                            children=html.Div(className='control-tab', children=[
                                html.H4(
                                    className='what-is',
                                    children='COVID-19 Risk Calculator'
                                ),
                                html.P(
                                    """
                                    This product uses proteomics, metabolomics, and clinical data to predict a COVID-19 severity response of healthy, mild, moderate, or severe. This product is targeted towards doctors, health professionals and researchers with access to patients’ omics data; however, the general public may also utilise this tool by inputting their clinical characteristics
                                    """
                                ),
                                
                            ])
                        ),
                        dcc.Tab(
                                    label='Protein',
                                    value='alignment-tab-select',
                                    children=html.Div(className='control-tab', children=[
                                        html.Div(className='app-controls-block', children=[
                                            html.Div("Patient information",style={'font-weight': 'bold', 'font-size': 14}
                                            ),
                                        ]),
                                        dbc.Row([
                                            dbc.Col(html.Div([
                                                html.Label('P05231: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='p_f1',
                                                    value=10.4968192328414
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('P08727: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='p_f2',
                                                    value=6.91532585383785
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('P80098: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='p_f3',
                                                    value=5.36527102042059
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('P15018: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='p_f4',
                                                    value=3.56843545068835
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('Q12933: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='p_f5',
                                                    value=2.49677049592325
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('P80511: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='p_f6',
                                                    value=4.3154520859424
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('O14867: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='p_f7',
                                                    value=2.18898700186335
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('Q07065: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='p_f8',
                                                    value=6.4383147470324
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('O95786: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='p_f9',
                                                    value=4.81102396829665
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('Q9NZQ7: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='p_f10',
                                                    value=5.9390800051505
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('P15514: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='p_f11',
                                                    value=4.4973716797173
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('Q92844: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='p_f12',
                                                    value=2.57114253736795
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('P14210: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='p_f13',
                                                    value=11.9822297060336
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('Q9UQV4: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='p_f14',
                                                    value=4.72449702980865
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('P02778: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='p_f15',
                                                    value=12.3284400278332
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('Q9P0M4: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='p_f16',
                                                    value=5.08938838995985
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('Q9C035: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='p_f17',
                                                    value=2.36324235488535
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('P51617: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='p_f18',
                                                    value=1.99519670541399
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('Q05516: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='p_f19',
                                                    value=3.2821636768204
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('P78362: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='p_f20',
                                                    value=1.55369628175705
                                                )
                                            ],style={"width": "50%"},), ),
                                            
                                            
                                        ]),
                                        html.Button(id='submit-button-p', n_clicks=0, children='Submit'),
                                        html.Div(id='output-state-p'),

                                    ])
                                ),
                        dcc.Tab(
                                    label='Metabolites',
                                    value='alignment-tab-select2',
                                    children=html.Div(className='control-tab', children=[
                                        html.Div(className='app-controls-block', children=[
                                            html.Div("Patient information",style={'font-weight': 'bold', 'font-size': 14}
                                            ),
                                        ]),
                                        dbc.Row([
                                            dbc.Col(html.Div([
                                                html.Label('1,2-dilinoleoyl-GPC (18:2/18:2): '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='m_f1',
                                                    value=0.479254962
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('1-(1-enyl-palmitoyl)-GPC (P-16:0)*: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='m_f2',
                                                    value=0.199539983999999
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('1-linoleoyl-GPC (18:2): '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='m_f3',
                                                    value=0.411358163999999
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('1-linoleoyl-GPE (18:2)*: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='m_f4',
                                                    value=0.516010309
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('1-palmitoyl-GPC (16:0): '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='m_f5',
                                                    value=0.604323368999999
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('1-stearoyl-GPC (18:0): '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='m_f6',
                                                    value=0.335182243
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('3-hydroxy-2-methylpyridine sulfate: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='m_f7',
                                                    value=1.7008984855
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('3-hydroxypyridine sulfate: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='m_f8',
                                                    value=0.039907569
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('3-methyl catechol sulfate (1): '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='m_f9',
                                                    value=0.066674519
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('3-ureidopropionate: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='m_f10',
                                                    value=1.4761153965
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('6-bromotryptophan: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='m_f11',
                                                    value=0.327419351
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('6-oxopiperidine-2-carboxylate: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='m_f12',
                                                    value=5.128455532
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('N-(2-furoyl)glycine: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='m_f13',
                                                    value=1.56077888099999
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('aspartate: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='m_f14',
                                                    value=1.214429453
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('beta-hydroxyisovalerate: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='m_f15',
                                                    value=2.47857508
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('cystine: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='m_f16',
                                                    value=0.745053342999999
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('eicosanedioate (C20-DC): '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='m_f17',
                                                    value=0.646271681
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('levulinate (4-oxovalerate): '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='m_f18',
                                                    value=0.183619287
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('mannose: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='m_f19',
                                                    value=3.875073509
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('sphingomyelin (d18:1/21:0, d17:1/22:0, d16:1/23:0)*: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='m_f20',
                                                    value=0.44791752
                                                )
                                            ],style={"width": "50%"},), ),
                                            
                                            
                                        ]),
                                        html.Button(id='submit-button-m', n_clicks=0, children='Submit'),
                                        html.Div(id='output-state-m'),



                                    ])
                                ),
                        dcc.Tab(
                                    label='Combine MLP',
                                    value='control-tab-select3',
                                    children=html.Div(className='control-tab', children=[
                                        html.Div(className='app-controls-block', children=[
                                            html.Div("Patient information",style={'font-weight': 'bold', 'font-size': 14}
                                            ),
                                        ]),
                                        dbc.Row([
                                            dbc.Col(html.Div([
                                                html.Label('P05231: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mp_f1',
                                                    value=10.4968192328414
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('P08727: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mp_f2',
                                                    value=6.91532585383785
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('P80098: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mp_f3',
                                                    value=5.36527102042059
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('P15018: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mp_f4',
                                                    value=3.56843545068835
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('Q12933: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mp_f5',
                                                    value=2.49677049592325
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('P80511: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mp_f6',
                                                    value=4.3154520859424
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('O14867: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mp_f7',
                                                    value=2.18898700186335
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('Q07065: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mp_f8',
                                                    value=6.4383147470324
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('O95786: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mp_f9',
                                                    value=4.81102396829665
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('Q9NZQ7: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mp_f10',
                                                    value=5.9390800051505
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('P15514: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mp_f11',
                                                    value=4.4973716797173
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('Q92844: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mp_f12',
                                                    value=2.57114253736795
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('P14210: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mp_f13',
                                                    value=11.9822297060336
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('Q9UQV4: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mp_f14',
                                                    value=4.72449702980865
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('P02778: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mp_f15',
                                                    value=12.3284400278332
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('Q9P0M4: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mp_f16',
                                                    value=5.08938838995985
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('Q9C035: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mp_f17',
                                                    value=2.36324235488535
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('P51617: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mp_f18',
                                                    value=1.99519670541399
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('Q05516: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mp_f19',
                                                    value=3.2821636768204
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('P78362: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mp_f20',
                                                    value=1.55369628175705
                                                )
                                            ],style={"width": "50%"},), ),
                                            
                                            
                                        ]),
                                        dbc.Row([
                                            dbc.Col(html.Div([
                                                html.Label('1,2-dilinoleoyl-GPC (18:2/18:2): '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mm_f1',
                                                    value=0.479254962
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('1-(1-enyl-palmitoyl)-GPC (P-16:0)*: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mm_f2',
                                                    value=0.199539983999999
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('1-linoleoyl-GPC (18:2): '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mm_f3',
                                                    value=0.411358163999999
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('1-linoleoyl-GPE (18:2)*: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mm_f4',
                                                    value=0.516010309
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('1-palmitoyl-GPC (16:0): '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mm_f5',
                                                    value=0.604323368999999
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('1-stearoyl-GPC (18:0): '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mm_f6',
                                                    value=0.335182243
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('3-hydroxy-2-methylpyridine sulfate: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mm_f7',
                                                    value=1.7008984855
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('3-hydroxypyridine sulfate: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mm_f8',
                                                    value=0.039907569
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('3-methyl catechol sulfate (1): '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mm_f9',
                                                    value=0.066674519
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('3-ureidopropionate: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mm_f10',
                                                    value=1.4761153965
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('6-bromotryptophan: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mm_f11',
                                                    value=0.327419351
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('6-oxopiperidine-2-carboxylate: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mm_f12',
                                                    value=5.128455532
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('N-(2-furoyl)glycine: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mm_f13',
                                                    value=1.56077888099999
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('aspartate: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mm_f14',
                                                    value=1.214429453
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('beta-hydroxyisovalerate: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mm_f15',
                                                    value=2.47857508
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('cystine: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mm_f16',
                                                    value=0.745053342999999
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('eicosanedioate (C20-DC): '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mm_f17',
                                                    value=0.646271681
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('levulinate (4-oxovalerate): '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mm_f18',
                                                    value=0.183619287
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('mannose: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mm_f19',
                                                    value=3.875073509
                                                )
                                            ],style={"width": "50%"},), ),
                                            dbc.Col(html.Div([
                                                html.Label('sphingomyelin (d18:1/21:0, d17:1/22:0, d16:1/23:0)*: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='mm_f20',
                                                    value=0.44791752
                                                )
                                            ],style={"width": "50%"},), ),
                                            
                                            
                                        ]),
                                        html.Button(id='submit-button-mlp', n_clicks=0, children='Submit'),
                                        html.Div(id='output-state-mlp'),

                                        # html.Div(
                                        #     className='app-controls-name',
                                        #     children='Event Metadata'
                                        # ),
                                        # html.P('Hover or click on data to see it here.'),
                                        # html.Div(
                                        #     id='alignment-events'
                                        # )
                                    ]),
                        ),
                        dcc.Tab(
                                    label='Clinical',
                                    value='alignment-tab-select4',
                                    children=html.Div(className='control-tab', children=[
                                        html.Div(className='app-controls-block', children=[
                                            html.Div("Patient information",style={'font-weight': 'bold', 'font-size': 14}
                                            ),
                                        ]),
                                    dbc.Row([
                                            dbc.Col(html.Div([
                                                html.Label('Sex: '),
                                        dcc.Dropdown(
                                            options=[
                                            {
                                                'label': "Male",
                                                'value': 'Male'
                                            },
                                            {
                                                'label': "Female",
                                                'value': 'Female'
                                            },
                                           
                                        ],
                                        # options=[{'label':name, 'value':name} for name in names],
                                        # value = list(fnameDict.keys())[0],
                                        id='c_f1',
                                        value= 'Female'
                                        )
                                    ]), width={'size':6}),
                                    ]),
                                    dbc.Row([
                                            dbc.Col(html.Div([
                                                html.Label('Age: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='c_f2',
                                                    value=77
                                                )
                                            ],style={"width": "47.5%"},), ),
                                    ]),
                                    dbc.Row([
                                            dbc.Col(html.Div([
                                                html.Label('BMI: '),
                                                dcc.Input(
                                                    type="number",
                                                    debounce=True,
                                                    id='c_f3',
                                                    value=33.657783
                                                )
                                            ],style={"width": "47.5%"},), ),
                                    ]),
                                    dbc.Row([
                                            dbc.Col(html.Div([
                                                html.Label('Cigarette Smoking: '),
                                        dcc.Dropdown(
                                            options=[
                                            {
                                                'label': "Never",
                                                'value': 'Never'
                                            },
                                            {
                                                'label': "Former",
                                                'value': 'Former'
                                            },
                                            {
                                                'label': "Current",
                                                'value': 'Current'
                                            }
                                           
                                        ],
                                        id='c_f4',
                                        value= 'Never'
                                        )
                                    ]), width={'size':6}),
                                    ]),
                                    dbc.Row([
                                            dbc.Col(html.Div([
                                                html.Label('Diabetes: '),
                                        dcc.Dropdown(
                                            options=[
                                            {
                                                'label': "No",
                                                'value': 'No'
                                            },
                                            {
                                                'label': "T1DM",
                                                'value': 'T1DM'
                                            },
                                            {
                                                'label': "T2DM",
                                                'value': 'T2DM'
                                            }
                                           
                                        ],
                                        id='c_f5',
                                        value= 'No'
                                        )
                                    ]), width={'size':6}),
                                    ]),
                                    dbc.Row([
                                            dbc.Col(html.Div([
                                                html.Label('Asthma: '),
                                        dcc.Dropdown(
                                            options=[
                                            {
                                                'label': "No",
                                                'value': 'No'
                                            },
                                            {
                                                'label': "Yes",
                                                'value': 'Yes'
                                            },
                                           
                                        ],
                                        id='c_f6',
                                        value= 'No'
                                        )
                                    ]), width={'size':6}),
                                    ]),
                                    dbc.Row([
                                            dbc.Col(html.Div([
                                                html.Label('Cancer: '),
                                        dcc.Dropdown(
                                            options=[
                                            {
                                                'label': "No",
                                                'value': 'No'
                                            },
                                            {
                                                'label': "Yes",
                                                'value': 'Yes'
                                            },
                                           
                                        ],
                                        id='c_f7',
                                        value= 'No'
                                        )
                                    ]), width={'size':6}),
                                    ]),
                                    dbc.Row([
                                            dbc.Col(html.Div([
                                                html.Label('Chronic Hypertension: '),
                                        dcc.Dropdown(
                                            options=[
                                            {
                                                'label': "No",
                                                'value': 'No'
                                            },
                                            {
                                                'label': "Yes",
                                                'value': 'Yes'
                                            },
                                           
                                        ],
                                        id='c_f8',
                                        value= 'Yes'
                                        )
                                    ]), width={'size':6}),
                                    ]),
                                    dbc.Row([
                                            dbc.Col(html.Div([
                                                html.Label('Chronic Kidney Disease: '),
                                        dcc.Dropdown(
                                            options=[
                                            {
                                                'label': "No",
                                                'value': 'No'
                                            },
                                            {
                                                'label': "Yes",
                                                'value': 'Yes'
                                            },
                                           
                                        ],
                                        id='c_f9',
                                        value= 'No'
                                        )
                                    ]), width={'size':6}),
                                    ]),
                                    dbc.Row([
                                            dbc.Col(html.Div([
                                                html.Label('Congestive Heart Failure: '),
                                        dcc.Dropdown(
                                            options=[
                                            {
                                                'label': "No",
                                                'value': 'No'
                                            },
                                            {
                                                'label': "Yes",
                                                'value': 'Yes'
                                            },
                                           
                                        ],
                                        id='c_f10',
                                        value= 'No'
                                        )
                                    ]), width={'size':6}),
                                    ]),
                                    dbc.Row([
                                            dbc.Col(html.Div([
                                                html.Label('COPD: '),
                                        dcc.Dropdown(
                                            options=[
                                            {
                                                'label': "No",
                                                'value': 'No'
                                            },
                                            {
                                                'label': "Yes",
                                                'value': 'Yes'
                                            },
                                           
                                        ],
                                        id='c_f11',
                                        value= 'No'
                                        )
                                    ]), width={'size':6}),
                                    ]),
                                    dbc.Row([
                                            dbc.Col(html.Div([
                                                html.Label('Coronary Artery Disease: '),
                                        dcc.Dropdown(
                                            options=[
                                            {
                                                'label': "No",
                                                'value': 'No'
                                            },
                                            {
                                                'label': "Yes",
                                                'value': 'Yes'
                                            },
                                           
                                        ],
                                        id='c_f12',
                                        value= 'No'
                                        )
                                    ]), width={'size':6}),
                                    ]),
                                            
                                            
                                            
                                        # ]),
                                        html.Button(id='submit-button-c', n_clicks=0, children='Submit'),
                                        html.Div(id='output-state-c'),

                                    ])
                                ),
                        
                        ]),
                    ],style={'display': 'inline-block'}),
                                html.Div(id='plots', className='plots', children=[
                                    dbc.Row(
                                    html.Div(
                                        className='Severity',
                                        children='Prediction Description',
                                        style={'font-weight': 'bold', 'font-size': 20}
                                    )),
                                    dbc.Row([
                                        html.Div(
                                            dash_table.DataTable(
                                                data=df.to_dict('records'),
                                                sort_action='native',
                                                columns=[
                                                    {'name': 'Our classification', 'id': 'Our classification', 'type': 'text'},
                                                    {'name': 'Description', 'id': 'Description', 'type': 'text'},
                                                    {'name': 'WHO Oridinal Scale', 'id': 'WHO Oridinal Scale', 'type': 'numeric'},
                                                ],
                                                editable=False,
                                                style_data_conditional=[
                                                    {
                                                        'if': {
                                                            'filter_query': '{Our classification} = "Severe"'
                                                        },
                                                        'backgroundColor': '#F1948A',
                                                        'color': 'black'
                                                    },
                                                    {
                                                        'if': {
                                                            'filter_query': '{Our classification} = "Mild"'
                                                        },
                                                        'backgroundColor': '#F9E79F',
                                                        'color': 'black'
                                                    },
                                                    {
                                                        'if': {
                                                            'filter_query': '{Our classification} = "Moderate"'
                                                        },
                                                        'backgroundColor': '#F5CBA7',
                                                        'color': 'black'
                                                    },
                                                    {
                                                        'if': {
                                                            'filter_query': '{Our classification} = "Healthy"'
                                                        },
                                                        'backgroundColor': '#ABEBC6',
                                                        'color': 'black'
                                                    }
                                                ],
                                                style_header={
                                                    'backgroundColor': 'rgb(230, 230, 230)',
                                                    'fontWeight': 'bold'
                                                }
                                                )
                                        )
                                    
                                ]),
                                dbc.Row(
                                    html.Div(
                                        className='Comparison',
                                        children='Comparison of Different Groups',
                                        style={'font-weight': 'bold', 'font-size': 20}
                                    )),
                                dbc.Row([
                                    dbc.Col(html.Div([
                                        html.Label('Choose a Dataset: '),
                                        dcc.Dropdown(
                                            
                                        options=[{'label':name, 'value':name} for name in names],
                                        value = list(fnameDict.keys())[0],
                                        id='dataset'
                                        )
                                    ]), width={'size':4}),
                                    dbc.Col(html.Div([
                                        html.Label('Choose a Feature: '),
                                        dcc.Dropdown(
                                            # options=p_options,
                                            value=fnameDict[list(fnameDict.keys())[0]][0],
                                            id='p_ids'
                                        )
                                    ]), width={'size':5}),
                                ]),
                                dbc.Row([
                                    html.Div(
                                        dcc.Graph(
                                            id='boxplot'
                                        )
                                    )
                                    
                                ]),
                                        
                                        
                                    
                                    
                                 
                                ],style={"margin-left": "55px",'display': 'inline-block'}),
                    
                
            # ]),
        ]),
    ])


def callbacks(_app):
    @_app.callback(Output('output-state-p', 'children'),
              [Input('submit-button-p', 'n_clicks')],
              [Input('p_f1', 'value'),
               Input('p_f2', 'value'),
               Input('p_f3', 'value'),
               Input('p_f4', 'value'),
               Input('p_f5', 'value'),
               Input('p_f6', 'value'),
               Input('p_f7', 'value'),
               Input('p_f8', 'value'),
               Input('p_f9', 'value'),
               Input('p_f10', 'value'),
               Input('p_f11', 'value'),
               Input('p_f12', 'value'),
               Input('p_f13', 'value'),
               Input('p_f14', 'value'),
               Input('p_f15', 'value'),
               Input('p_f16', 'value'),
               Input('p_f17', 'value'),
               Input('p_f18', 'value'),
               Input('p_f19', 'value'),
               Input('p_f20', 'value')
               ])
    def update_output_p(n_clicks, p_f1, p_f2, p_f3, p_f4, p_f5, p_f6, p_f7, p_f8, p_f9, p_f10, p_f11, p_f12, p_f13, p_f14, p_f15, p_f16, p_f17, p_f18, p_f19, p_f20):
        # global _input
        # _input.append(input1)
        if n_clicks == 0:
            return "Please fill in Proteins and click Submit to predict."
        _input = [p_f1, p_f2, p_f3, p_f4, p_f5, p_f6, p_f7, p_f8, p_f9, p_f10, p_f11, p_f12, p_f13, p_f14, p_f15, p_f16, p_f17, p_f18, p_f19, p_f20]
        _input = np.array(_input)
        _input = _input.reshape(1, -1)
        # load the scaler
        input_scaled = scaler_protein.transform(_input)
        # global clf
        _pred = str(protein_clf.predict(input_scaled)[0])
        return (html.P(["Your Prediction is",html.Br(),"{}".format(_pred),html.Br(),"This is prediction number {}.".format(n_clicks)]))


    @_app.callback(Output('output-state-m', 'children'),
              [Input('submit-button-m', 'n_clicks')],
              [Input('m_f1', 'value'),
               Input('m_f2', 'value'),
               Input('m_f3', 'value'),
               Input('m_f4', 'value'),
               Input('m_f5', 'value'),
               Input('m_f6', 'value'),
               Input('m_f7', 'value'),
               Input('m_f8', 'value'),
               Input('m_f9', 'value'),
               Input('m_f10', 'value'),
               Input('m_f11', 'value'),
               Input('m_f12', 'value'),
               Input('m_f13', 'value'),
               Input('m_f14', 'value'),
               Input('m_f15', 'value'),
               Input('m_f16', 'value'),
               Input('m_f17', 'value'),
               Input('m_f18', 'value'),
               Input('m_f19', 'value'),
               Input('m_f20', 'value')
               ])
    def update_output_m(n_clicks, m_f1, m_f2, m_f3, m_f4, m_f5, m_f6, m_f7, m_f8, m_f9, m_f10, m_f11, m_f12, m_f13, m_f14, m_f15, m_f16, m_f17, m_f18, m_f19, m_f20):
        # global _input
        # _input.append(input1)
        if n_clicks == 0:
            return "Please fill in Metabolites and click Submit to predict."
        _input = [m_f1, m_f2, m_f3, m_f4, m_f5, m_f6, m_f7, m_f8, m_f9, m_f10, m_f11, m_f12, m_f13, m_f14, m_f15, m_f16, m_f17, m_f18, m_f19, m_f20]
        _input = np.array(_input)
        _input = _input.reshape(1, -1)
        # load the scaler
        input_scaled = scaler_meta.transform(_input)
        # global clf
        _pred = meta_clf.predict(input_scaled)[0]
        label = ['Healthy', 'Mild', 'Moderate', 'Severe']
        return (html.P(["Your Prediction is",html.Br(),"{}".format(label[_pred]),html.Br(),"This is prediction number {}.".format(n_clicks)]))


    @_app.callback(Output('output-state-mlp', 'children'),
              [Input('submit-button-mlp', 'n_clicks')],
              [Input('mp_f1', 'value'),
               Input('mp_f2', 'value'),
               Input('mp_f3', 'value'),
               Input('mp_f4', 'value'),
               Input('mp_f5', 'value'),
               Input('mp_f6', 'value'),
               Input('mp_f7', 'value'),
               Input('mp_f8', 'value'),
               Input('mp_f9', 'value'),
               Input('mp_f10', 'value'),
               Input('mp_f11', 'value'),
               Input('mp_f12', 'value'),
               Input('mp_f13', 'value'),
               Input('mp_f14', 'value'),
               Input('mp_f15', 'value'),
               Input('mp_f16', 'value'),
               Input('mp_f17', 'value'),
               Input('mp_f18', 'value'),
               Input('mp_f19', 'value'),
               Input('mp_f20', 'value'),
               Input('mm_f1', 'value'),
               Input('mm_f2', 'value'),
               Input('mm_f3', 'value'),
               Input('mm_f4', 'value'),
               Input('mm_f5', 'value'),
               Input('mm_f6', 'value'),
               Input('mm_f7', 'value'),
               Input('mm_f8', 'value'),
               Input('mm_f9', 'value'),
               Input('mm_f10', 'value'),
               Input('mm_f11', 'value'),
               Input('mm_f12', 'value'),
               Input('mm_f13', 'value'),
               Input('mm_f14', 'value'),
               Input('mm_f15', 'value'),
               Input('mm_f16', 'value'),
               Input('mm_f17', 'value'),
               Input('mm_f18', 'value'),
               Input('mm_f19', 'value'),
               Input('mm_f20', 'value')
               ])
    def update_output_m(n_clicks, mp_f1, mp_f2, mp_f3, mp_f4, mp_f5, mp_f6, mp_f7, mp_f8, mp_f9, mp_f10, mp_f11, mp_f12, mp_f13, mp_f14, mp_f15, mp_f16, mp_f17, mp_f18, mp_f19, mp_f20, mm_f1, mm_f2, mm_f3, mm_f4, mm_f5, mm_f6, mm_f7, mm_f8, mm_f9, mm_f10, mm_f11, mm_f12, mm_f13, mm_f14, mm_f15, mm_f16, mm_f17, mm_f18, mm_f19, mm_f20):
        # global _input
        # _input.append(input1)
        if n_clicks == 0:
            return "Please fill in Protein|Metabolites and click Submit to predict."
        _input_p = [mp_f1, mp_f2, mp_f3, mp_f4, mp_f5, mp_f6, mp_f7, mp_f8, mp_f9, mp_f10, mp_f11, mp_f12, mp_f13, mp_f14, mp_f15, mp_f16, mp_f17, mp_f18, mp_f19, mp_f20]
        _input_p = np.array(_input_p)

        _input_m = [mm_f1, mm_f2, mm_f3, mm_f4, mm_f5, mm_f6, mm_f7, mm_f8, mm_f9, mm_f10, mm_f11, mm_f12, mm_f13, mm_f14, mm_f15, mm_f16, mm_f17, mm_f18, mm_f19, mm_f20]
        _input_m = np.array(_input_m)

        _input_p = np.expand_dims(_input_p, 0)
        _input_m = np.expand_dims(_input_m, 0)

        _input_p = mlp_dataset_obj.mms_p.transform(_input_p)
        _input_m = mlp_dataset_obj.mms_m.transform(_input_m)

        _input_p = torch.tensor(_input_p)
        _input_m = torch.tensor(_input_m)

        output = mlp_model(_input_m.float(), _input_p.float())
        _, predicted = torch.max(output.data, 1)
        
        
        label = ['Healthy', 'Mild', 'Moderate', 'Severe']
        _pred = label[predicted.numpy()[0]]
        return (html.P(["Your Prediction is",html.Br(),"{}".format(_pred),html.Br(),"This is prediction number {}.".format(n_clicks)]))
    
    @_app.callback(Output('output-state-c', 'children'),
              [Input('submit-button-c', 'n_clicks')],
              [Input('c_f1', 'value'),
               Input('c_f2', 'value'),
               Input('c_f3', 'value'),
               Input('c_f4', 'value'),
               Input('c_f5', 'value'),
               Input('c_f6', 'value'),
               Input('c_f7', 'value'),
               Input('c_f8', 'value'),
               Input('c_f9', 'value'),
               Input('c_f10', 'value'),
               Input('c_f11', 'value'),
               Input('c_f12', 'value'),

               ])
            
    def update_output_c(n_clicks, c_f1, c_f2, c_f3, c_f4, c_f5, c_f6, c_f7, c_f8, c_f9, c_f10, c_f11, c_f12):
        if n_clicks == 0:
            return "Please fill in Clinical information and click Submit to predict."
        
        if c_f1 == 'Male':
            _c_f1 = 1
        else:
            _c_f1 = 0
        
        # _c_f3 = c_f3.reshape(1,-1)

        if c_f4 == 'Never':
            _c_f4 = 0
        elif c_f4 == 'Former':
            _c_f4 = 1
        else:
            _c_f4 = 2

        if c_f5 == 'No':
            _c_f5 = 0
        elif c_f5 == 'T1DM':
            _c_f5 = 1
        else:
            _c_f5 = 2

        if c_f6 == 'No':
            _c_f6 = 0
        else:
            _c_f6 = 1

        if c_f7 == 'No':
            _c_f7 = 0
        else:
            _c_f7 = 1

        if c_f8 == 'No':
            _c_f8 = 0
        else:
            _c_f8 = 1

        if c_f9 == 'No':
            _c_f9 = 0
        else:
            _c_f9 = 1

        if c_f10 == 'No':
            _c_f10 = 0
        else:
            _c_f10 = 1

        if c_f11 == 'No':
            _c_f11 = 0
        else:
            _c_f11 = 1

        if c_f12 == 'No':
            _c_f12 = 0
        else:
            _c_f12 = 1

        _input = [_c_f1, c_f2, c_f3, _c_f4, _c_f5, _c_f6, _c_f7, _c_f8, _c_f9, _c_f10, _c_f11, _c_f12]
        _input = np.array(_input)
        _input = _input.reshape(1, -1)

        # global clf
        _pred = str(clinical_clf.predict(_input)[0])
        return (html.P(["Your Prediction is",html.Br(),"{}".format(_pred),html.Br(),"This is prediction number {}.".format(n_clicks)]))

    @_app.callback(Output('boxplot', 'figure'),
                [Input('dataset','value')],
                [Input('p_ids', 'value')]
    )
    def box_plot(dataset, p_ids):
        data = DATASETS[dataset]
        ls=['Label']
        ls.append(p_ids)
        if p_ids not in data:
            fig = px.box(data, x='Label',color="Label")
        # data = dataset
        else:
            df = data[ls]
            fig = px.box(df, x='Label', y=p_ids,color="Label")
        return fig

    @_app.callback(
        Output('p_ids', 'options'),
        [Input('dataset', 'value')]
    )
    def update_date_dropdown(name):
        return [{'label': i, 'value': i} for i in fnameDict[name]]

app = run_standalone_app(layout, callbacks, header_colors, __file__)
server = app.server

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)