#
# MIT License
#
# Copyright (c) 2024 GT4SD team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import warnings

import importlib_resources

from enzeptional import (
    EnzymeOptimizer,
    SequenceMutator,
    SequenceScorer,
    CrossoverGenerator,
    HuggingFaceEmbedder,
    HuggingFaceModelLoader,
    HuggingFaceTokenizerLoader,
    SelectionGenerator,
)

warnings.simplefilter(action="ignore", category=FutureWarning)

scorer_filepath = (
    importlib_resources.files('enzeptional.resources.kcat_sample_model') / 'model.pkl'
)
scaler_filepath = (
    importlib_resources.files('enzeptional.resources.kcat_sample_model') / 'scaler.pkl'
)


def main(cro,num,seed):
    language_model_path = "facebook/esm2_t33_650M_UR50D"
    tokenizer_path = "facebook/esm2_t33_650M_UR50D"
    chem_model_path = "seyonec/ChemBERTa-zinc-base-v1"
    chem_tokenizer_path = "seyonec/ChemBERTa-zinc-base-v1"

    model_loader = HuggingFaceModelLoader()
    tokenizer_loader = HuggingFaceTokenizerLoader()

    protein_model = HuggingFaceEmbedder(
        model_loader=model_loader,
        tokenizer_loader=tokenizer_loader,
        model_path=language_model_path,
        tokenizer_path=tokenizer_path,
        cache_dir=None,
        device="cpu",
    )

    chem_model = HuggingFaceEmbedder(
        model_loader=model_loader,
        tokenizer_loader=tokenizer_loader,
        model_path=chem_model_path,
        tokenizer_path=chem_tokenizer_path,
        cache_dir=None,
        device="cpu",
    )

    mutation_config = {
        "type": "language-modeling",
        "embedding_model_path": language_model_path,
        "tokenizer_path": tokenizer_path,
        "unmasking_model_path": language_model_path,
    }

    intervals = [(12,35)]

    batch_size = 2
    top_k = 1

    substrate_smiles = "C(C(=O)C(=O)O)C(=O)O"
    product_smiles = "CC(=O)C(=O)[O-]"
    #6IVA
    sample_sequence = "MGSSHHHHHHSSGLVPRGSHMTNAALLLGEGFTLMFLGMGFVLAFLFLLIFAIRGMSAAVNRFFPEPAPAPKAAPAAAAPVVDDFTRLKPVIAAAIHHHHRLNA"

    mutator = SequenceMutator(sequence=sample_sequence, mutation_config=mutation_config)
    mutator.set_top_k(top_k)

    scorer = SequenceScorer(
        protein_model=protein_model,
        scorer_filepath=scorer_filepath,
        use_xgboost=True,
        scaler_filepath=scaler_filepath,
    )

    selection_generator = SelectionGenerator()
    crossover_generator = CrossoverGenerator()

    optimizer = EnzymeOptimizer(
        sequence=sample_sequence,
        mutator=mutator,
        scorer=scorer,
        intervals=intervals,
        substrate_smiles=substrate_smiles,
        product_smiles=product_smiles,
        chem_model=chem_model,
        selection_generator=selection_generator,
        crossover_generator=crossover_generator,
        concat_order=["substrate", "sequence"],
        batch_size=batch_size,
        selection_ratio=0.5,
        perform_crossover=True,
        selection_method="top",
        crossover_method=cro,
        use_intervals=False,
        pad_intervals=False,
        minimum_interval_length=8,
        seed=seed
    )

    num_iterations = 10
    num_sequences = 12
    num_mutations = num

    optimized_sequences = optimizer.optimize(
        num_iterations=num_iterations,
        num_sequences=num_sequences,
        num_mutations=num_mutations
    )

    return optimized_sequences

if __name__ == "__main__":
    num_list = [5,10,15]
    crossover_methods = ["sp", "uniform"]
    seeds = [36,91,144]
    for seed in seeds:
        for num_mutations in num_list:
            for cro in  crossover_methods:
                main(cro, num_mutations,seed)




