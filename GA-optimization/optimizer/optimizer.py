from typing import Any, Dict, Tuple, List, Optional
from optimizer.core import SequenceMutatorInitializer
from optimizer.fitness import FitnessEvalutor
from enzeptional import (HuggingFaceEmbedder)



class GeneticAlgorithmSelection():
    
    """
        Selecting top sequences based on their scores.

    """

    def selection(
        self,
        pool_of_sequences: List[Dict[str, Any]],
        k: float = 0.8,
    ) -> List[Any]:
        
        """Selects a subset of sequences from a pool based on their scores.

        Args:
            pool_of_sequences: A list of
            dictionaries, each containing a sequence and its score.
            k: A fraction representing the proportion of top sequences to select. Defaults to 0.8.

        Returns:
            A list of the top k sequences based on scores.
        """
    

class GeneticAlgorithmCrossover():
    
    """
        Perform crossover operations between sequences.
    
    """

    def sp_crossover(self, a_sequence: str, another_sequence: str) -> Tuple[str, str]:
        
        """Performs a single point crossover between two sequences.

        Args:
            a_sequence: The first sequence for crossover.
            another_sequence: The second sequence for crossover.

        Returns:
            A tuple of two new sequences resulting
            from the crossover.
        """

    # Also think of other ways to execute this apart form SP? Uniform crossover?
    # Crossover that doesnt make sure that the protein structure is compromised?


class GeneticOptimizer:
    """
    Optimizes protein sequences based on interaction with
    substrates and products.
    """

    def __init__(
        self,
        sequence: str,
        mutator: SequenceMutatorInitializer,
        scorer: FitnessEvalutor,
        intervals: List[Tuple[int, int]],
        substrate_smiles: str,
        product_smiles: str,
        chem_model: HuggingFaceEmbedder,
        selection_generator: GeneticAlgorithmSelection,
        crossover_generator: GeneticAlgorithmCrossover,
        concat_order: List[str],
        batch_size: int = 2,
        selection_ratio: float = 0.5,
        selection_method: str = "top",
        perform_crossover: bool = False,
        crossover_method: str = "sp",
        use_intervals=False,
        minimum_interval_length: int = 8,
        pad_intervals: bool = False,
        seed: int = 123,
    ):
        """Initializes the optimizer with models, sequences, and
        optimization parameters.


        Args:
            sequence: The initial protein sequence.
            protein_model: Model for protein embeddings.
            substrate_smiles: SMILES representation of the substrate.
            product_smiles: SMILES representation of the product.
            chem_model_path: Path to the chemical model.
            chem_tokenizer_path: Path to the chemical tokenizer.
            scorer_filepath: File path to the scoring model.
            mutator: The mutator for generating sequence variants.
            intervals: Intervals for mutation.
            batch_size: The number of sequences to process in one batch.
            top_k: Number of top mutations to consider.
            selection_ratio: Ratio of sequences to select after scoring.
            perform_crossover: Flag to perform crossover operation.
            crossover_type: Type of crossover operation.
            minimum_interval_length: Minimum length of mutation intervals.
            pad_intervals: Flag to pad the intervals.
            concat_order: Order of concatenating embeddings.
            scaler_filepath: Path to the scaler in case you are using the Kcat model.
            use_xgboost_scorer: Flag to specify if the fitness function is the Kcat.
        """
        self.sequence = sequence
        self.mutator = mutator
        self.scorer = scorer
        self.intervals = (
            sanitize_intervals(intervals) if intervals else [(0, len(sequence))]#合并区间
        )
        self.batch_size = batch_size
        self.selection_ratio = selection_ratio
        self.selection_method = selection_method
        self.perform_crossover = perform_crossover
        self.crossover_method = crossover_method
        self.use_intervals = use_intervals
        self.concat_order = concat_order
        self.seed = seed
        random.seed(self.seed)

        self.chem_model = chem_model
        self.substrate_embedding = chem_model.embed([substrate_smiles])[0]
        self.product_embedding = chem_model.embed([product_smiles])[0]
        self.selection_generator = selection_generator
        self.crossover_generator = crossover_generator

        if pad_intervals:
            self.intervals = sanitize_intervals_with_padding(
                self.intervals, minimum_interval_length, len(sequence)
            )

        self.all_mutated_sequences = set([self.sequence])

    def optimize(
        self,
        num_iterations: int,
        num_sequences: int,
        num_mutations: int,
        time_budget: Optional[int] = 360,
    ) -> List[str]:
        """Runs the optimization process over a specified number of iterations.

        Args:
            num_iterations: Number of iterations to run the optimization.
            num_sequences: Number of sequences to generate per iteration.
            num_mutations: Max number of mutations to apply.
            time_budget (Optional[int]): Time budget for optimizer (in seconds). Defaults to 360.

        Returns:
            A tuple containing the list of all sequences and iteration information (all_scored_sequences).
        """


    def _perform_crossover(self, selected_sequences: List[Dict[str, Any]]) -> Tuple[str]:

        """
        Performs crossover on the selected sequences. 

        
        Args:
            selected_sequences: tuple containing the list of all sequences that could be treated as parents.

        Returns:
            Tuple[str] or List[str]: Child sequences.

        """
