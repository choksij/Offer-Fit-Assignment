import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import pytest
from execution import DataModeler

@pytest.fixture
def sample_dm():
    df = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "amount": [10, np.nan, 5],
        "transaction_date": ['2022-01-01', None, '2022-03-01'],
        "outcome": [True, False, True]
    })
    dm = DataModeler(df)
    return dm

def test_prepare_and_impute(sample_dm):
    dm = sample_dm
    prepared = dm.prepare_data()
    assert prepared.dtypes.to_dict() == {                            # both columns present and float
        'amount': np.dtype('float64'),
        'transaction_date': np.dtype('float64'),
    }
    imputed = dm.impute_missing()
    assert not imputed.isna().any().any()

def test_fit_summary_and_predict(sample_dm):
    dm = sample_dm
    dm.prepare_data()
    dm.impute_missing()
    dm.fit()
    summary = dm.model_summary()
    assert "Rule-based threshold model" in summary
    preds = dm.predict()
    assert isinstance(preds, np.ndarray) and preds.dtype == bool and preds.shape == (3,)            # Predictions must be boolean numpy array of length 3

def test_threshold_edge_case():
    df = pd.DataFrame({                                                 # exactly at mean should return True
        "customer_id": [1, 2],
        "amount": [5, 5],
        "transaction_date": ['2022-01-01', '2022-01-01'],
        "outcome": [True, True]
    })
    dm = DataModeler(df)
    dm.prepare_data()
    dm.impute_missing()
    dm.fit()
    preds = dm.predict()
    assert all(preds)

def test_save_load(tmp_path, sample_dm):
    dm = sample_dm
    dm.prepare_data()
    dm.impute_missing()
    dm.fit()
    pkl = tmp_path / "model.pkl"
    dm.save(str(pkl))
    loaded = DataModeler.load(str(pkl))
    assert dm.model_summary() == loaded.model_summary()                    # loaded summary should match
