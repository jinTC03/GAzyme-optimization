import warnings
import importlib_resources

from generator import ProteinLLM, ChemicalLLM
from optimizer.fitness import FitnessEvalutor
from optimizer.core import SequenceMutatorInitializer
from optimizer.Generator import NewSelectionGenerator, NewCrossoverGenerator

from enzeptional import (SelectionGenerator, CrossoverGenerator, 
    EnzymeOptimizer, HuggingFaceModelLoader, HuggingFaceTokenizerLoader)

warnings.simplefilter(action="ignore", category=FutureWarning)

#+ things to define 

# input sequence
# fitness model for evaluation 

def run_optimizer(sel,cro,num,seed):
    substrate_smiles = "C(C(=O)C(=O)O)C(=O)O"
    product_smiles = "CC(=O)C(=O)[O-]"
    #6IVA
    sample_sequence = "MGSSHHHHHHSSGLVPRGSHMTNAALLLGEGFTLMFLGMGFVLAFLFLLIFAIRGMSAAVNRFFPEPAPAPKAAPAAAAPVVDDFTRLKPVIAAAIHHHHRLNA"

    language_model_path = "facebook/esm2_t33_650M_UR50D"
    tokenizer_path = "facebook/esm2_t33_650M_UR50D"
    model_loader = HuggingFaceModelLoader()
    tokenizer_loader = HuggingFaceTokenizerLoader()
    chem_model_path = "seyonec/ChemBERTa-zinc-base-v1"
    chem_tokenizer_path = "seyonec/ChemBERTa-zinc-base-v1"

    # define protein generator 
    protein_model = ProteinLLM(model_loader, tokenizer_loader, language_model_path, tokenizer_path, device = "cpu")

    # define chembert model for embedding generation 
    scorer_filepath = (importlib_resources.files('enzeptional.resources.kcat_sample_model') / 'model.pkl')
    scaler_filepath = (
            importlib_resources.files('enzeptional.resources.kcat_sample_model') / 'scaler.pkl'
        )
    chem_model = ChemicalLLM(model_loader, tokenizer_loader, chem_model_path, chem_tokenizer_path,
        device="cpu")

    # define fitness model
    scorer = FitnessEvalutor(
        protein_model=protein_model,
        scorer_filepath=scorer_filepath,
        use_xgboost=True,
        scaler_filepath=scaler_filepath)


    # define configs for the genetic algorithm 
    # language-modelling part only contains masking residues at random places and unmasking using 
    # top K sampling 

    mutation_config = {
            "type": "language-modeling",
            "embedding_model_path": language_model_path,
            "tokenizer_path": tokenizer_path,
            "unmasking_model_path": language_model_path,
        }

    #define interval
    intervals = [(12,35)]

    batch_size = 2
    top_k = 1

    mutator = SequenceMutatorInitializer(sequence=sample_sequence, mutation_config=mutation_config)
    mutator.set_top_k(top_k)

    selection_generator = NewSelectionGenerator()
    crossover_generator = NewCrossoverGenerator()

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
        selection_method= sel,
        crossover_method= cro,
        use_intervals=True,
        perform_crossover=True,
        pad_intervals=False,
        minimum_interval_length=8,
        seed=seed,
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


def main():
    num_list = [5, 10, 15]
    selection_methods = ["top", "tournament", "roulette"]
    crossover_methods = ["sp", "tp", "uniform"]
    seeds = [36,91,144]
    for seed in seeds:
         for num_mutations in num_list:
            for sel, cro in zip(selection_methods, crossover_methods):
                run_optimizer(sel,cro,num_mutations,seed)


if __name__ == "__main__":
    main()

