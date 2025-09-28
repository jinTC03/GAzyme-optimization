# code borrowed from github repository https://github.com/GT4SD/enzeptional.git

import warnings
from typing import List, Dict, Optional, Any
import numpy as np
from joblib import load
import xgboost as xgb
import logging

from enzeptional import (
    SequenceScorer,
    HuggingFaceEmbedder,
)


warnings.simplefilter(action="ignore", category=FutureWarning)


class FitnessEvalutor():

    '''
    Wrapper class for GT4SD pre-trained Kcat and feasibility models. 
    Supports individual and batch scoring, with optional scaling and XGBoost support.

    Args: 
            protein_model: Model used for generating protein embeddings.
            scorer_filepath: Path to the trained model for scoring.
            use_xgboost: Whether to use XGBoost as the scoring model (default is False).
            scaler_filepath: Path to a scaler for feature normalization (optional).

    '''
    
    def __init__(self, protein_model: HuggingFaceEmbedder, 
                 scorer_filepath: str, use_xgboost: bool = False,
                 scaler_filepath: Optional[str] = None):

        if hasattr(protein_model, "generator") and hasattr(protein_model.generator, "embed"):
            embedder = protein_model.generator
        else:
            embedder = protein_model
        self.scorer = SequenceScorer(protein_model=embedder, scorer_filepath=scorer_filepath,
            use_xgboost=use_xgboost, scaler_filepath=scaler_filepath)

    def score(self, sequence, substrate_embedding, product_embedding, concat_order):
        return self.scorer.score(sequence, substrate_embedding, product_embedding, concat_order)

    def score_batch(self, sequences, substrate_embedding, product_embedding, concat_order):
        return self.scorer.score_batch(sequences, substrate_embedding, product_embedding, concat_order)