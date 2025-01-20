import os
from inspect_ai import Epochs, Task, task, eval
from inspect_ai.dataset import Sample, hf_dataset, MemoryDataset
from inspect_ai.scorer import Score, ScoreReducer, choice, score_reducer, Scorer, Target, INCORRECT
from inspect_ai.scorer._reducer.registry import score_reducer
from typing import Any, Literal, Optional

import torch

from scorers.scorers_rewording import JUDGE_FILTERED
from solvers.adaptive_utils import multiple_choice_save_cot
from solvers.solver_adaptive_truthfulqa import (
    adaptive_truthfulqa_solver,
    adaptive_truthfulqa_judge_solver
)
from solvers.solver_adaptive_truthfulqa_refactor import (
    adaptive_truthfulqa_solver_refactor,
)
from scorers.scorers_truthfulqa import (
    adaptive_truthfulqa_scorer,
    adaptive_truthfulqa_judge_scorer,
    adaptive_truthfulqa_scorer_judged,
    re_evaluation_scorer,
    press_truthfulqa_scorer,
)
from solvers.solver_press_truthfulqa import press_truthfulqa_solver
from solvers.solver_evaluate_adaptive_questions import evaluate_adaptive_truthfulqa_questions

from data.eval_log_processing import read_eval_log_async
from eval_dump_inspect import labels_to_positions
from utils_elicitation.novelty import novelty_filter_judged_only, novelty_filter_matrix_reducer

@score_reducer(name="sum_score")
def sum_score() -> ScoreReducer:
    """Compute sum of all scores, excluding JUDGE_FILTERED."""
    def reduce(scores: list[Score]) -> Score:
        values = [s.value for s in scores if s.value != JUDGE_FILTERED]
        if not values:
            return Score(value=0)
        return Score(value=len(values))
    return reduce

@score_reducer(name="final_diversity_score")
def final_diversity_score() -> ScoreReducer:
    """Computes the score of the questions, modulo the samples that are very similar in embedding space.
    The way we do this is to compute all pairwise cosine similarities. If two questions are very similar, we remove one of the questions. The reducing procedure is as follows:
    - If both were answered incorrectly and neither is similar to the others, we randomly sample one of them.
    - If both were answered incorrectly and one is similar to the others, we remove the one similar to the others.
    - If both were answered incorrectly and they are both similar to the others, 
    """
    def reduce(scores: list[Score]) -> Score:
        values = [s.value for s in scores if s.value != JUDGE_FILTERED]
        if not values:
            return Score(value=0)
        return Score(value=len(values))
    return reduce

@task
def truthfulqa_initial(target: Literal["mc1", "mc2"] = "mc1", use_cot: bool = False, debug: bool = False) -> Task:
    """
    TruthfulQA: Measuring How Models Mimic Human Falsehoods

    Stephanie Lin, Jacob Hilton, Owain Evans
    https://arxiv.org/abs/2109.07958v2

    # Eval truthful QA with reference questions (mc1 - single true answer)
    inspect eval truthfulqa.py

    # Eval against the control questions (mc2 - multi-true answer)
    inspect eval truthfulqa.py -T target=mc2
    Inspect Task implementation for the TruthfulQA benchmark

    Args:
        target (Literal["mc1", "mc2"]): Whether to use the mc1 or mc2 targets
    """

    def record_to_sample(record: dict[str, Any]) -> Sample:
        return Sample(
            input=record["question"],
            choices=record[f"{target}_targets"]["choices"],
            target=labels_to_positions(record[f"{target}_targets"]["labels"]),
            metadata={"embedding": record["embedding"]},
        )

    dataset = hf_dataset(
        path="davisrbr/truthfulqa-embeddings",
        name="default",
        sample_fields=record_to_sample,
        split="validation",
        auto_id=True,
        shuffle=True,
    )
    if target == "mc1":
        multiple_correct = False
    else:
        multiple_correct = True
    if debug:
        dataset = dataset[:10]
    return Task(
        dataset=dataset,
        solver=[multiple_choice_save_cot(multiple_correct=multiple_correct, shuffle=True, cot=use_cot)],
        scorer=choice(),
    )

@task
def adaptive_truthfulqa(
    initial_log_path: str,
    n_positive_samples: int = 8,
    n_negative_samples: int = 8,
    generator_model_name: str = "openai/gpt-4o-mini",
    eval_model_name: str = "openai/gpt-4o-mini",
    self_check_model_name: Optional[str] = None,
    target: Literal["mc1", "mc2"] = "mc1",
    use_embeddings: bool = False,
    embeddings_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    similarity_threshold: float = 0.8,
    score_threshold: int = 4,
    max_attempts: int = 5,
    randomize_sampling: bool = False,
    cot_in_context: bool = False,
    use_cot_generator: bool = False,
    use_cot_evaluator: bool = False,
    original_eval_model_name: Optional[str] = None,
    judge_model_name: Optional[str] = None,
    use_eval_model_for_checker: bool = False,
) -> Task:
    """
    Adaptive TruthfulQA task that generates new questions based on model errors and evaluates the model on them.
    Optionally uses a separate judge model to evaluate truthfulness and validity of generated questions.
    """
    question_embedding_map = None
    if use_embeddings:
        def record_to_sample(record: dict[str, Any]) -> Sample:
            return Sample(
                input=record["question"],
                choices=record[f"{target}_targets"]["choices"],
                target=labels_to_positions(record[f"{target}_targets"]["labels"]),
                metadata={"embedding": record["embedding"]},
            )
        ds = hf_dataset(
            path="davisrbr/truthfulqa-embeddings",
            name="default",
            sample_fields=record_to_sample,
            split="validation",
            auto_id=True,
            shuffle=True,
        )

        # Build a mapping from question text to embeddings
        question_embedding_map = {
            s.input: torch.tensor(s.metadata["embedding"]) for s in ds
        }

    # Construct the solver list: the main adaptive solver plus (if specified) a judge solver
    solver_list = [
        adaptive_truthfulqa_solver(
            initial_log_path=initial_log_path,
            n_positive_samples=n_positive_samples,
            n_negative_samples=n_negative_samples,
            generator_model_name=generator_model_name,
            eval_model_name=eval_model_name,
            self_check_model_name=self_check_model_name,
            target=target,
            use_embeddings=use_embeddings,
            embeddings_model_name=embeddings_model_name,
            question_embedding_map=question_embedding_map,
            similarity_threshold=similarity_threshold,
            score_threshold=score_threshold,
            max_attempts=max_attempts,
            randomize_sampling=randomize_sampling,
            cot_in_context=cot_in_context,
            use_cot_generator=use_cot_generator,
            use_cot_evaluator=use_cot_evaluator,
            original_eval_model_name=original_eval_model_name,
            use_eval_model_for_checker=use_eval_model_for_checker,
        ),
    ]

    # If we have a separate judge model, append the judge solver
    if judge_model_name:
        solver_list.append(
            adaptive_truthfulqa_judge_solver(
                initial_log_path=initial_log_path,
                judge_model_name=judge_model_name,
            )
        )

    # Build up the scorers:
    scorer_list = [
        adaptive_truthfulqa_scorer(),
    ]
    # If we have a judge model, also include judge-based scorers
    if judge_model_name:
        scorer_list.append(adaptive_truthfulqa_judge_scorer())
        scorer_list.append(adaptive_truthfulqa_scorer_judged())

    return Task(
        dataset=MemoryDataset(name="adaptive_truthfulqa", samples=[]),
        solver=solver_list,
        scorer=scorer_list,
    )

@task  
def adaptive_truthfulqa_refactor(
    initial_log_path: str,
    n_positive_samples: int = 8,
    n_negative_samples: int = 8,
    generator_model_name: str = "openai/gpt-4o-mini",
    eval_model_name: str = "openai/gpt-4o-mini",
    self_check_model_name: Optional[str] = None,
    target: Literal["mc1", "mc2"] = "mc1",
    use_embeddings: bool = False,
    embeddings_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    similarity_threshold: float = 0.6,
    score_threshold: int = 4,
    max_attempts: int = 5,
    randomize_sampling: bool = False,
    cot_in_context: bool = False,
    use_cot_generator: bool = False,
    use_cot_evaluator: bool = False,
    original_eval_model_name: Optional[str] = None,
    judge_model_name: Optional[str] = None,
    use_eval_model_for_checker: bool = False,
) -> Task:
    """
    Adaptive TruthfulQA (refactored) task that generates new questions based on model errors and evaluates the model on them.
    """
    question_embedding_map = None
    if use_embeddings:
        def record_to_sample(record: dict[str, Any]) -> Sample:
            return Sample(
                input=record["question"],
                choices=record[f"{target}_targets"]["choices"],
                target=labels_to_positions(record[f"{target}_targets"]["labels"]),
                metadata={"embedding": record["embedding"]},
            )
        ds = hf_dataset(
            path="davisrbr/truthfulqa-embeddings",
            name="default",
            sample_fields=record_to_sample,
            split="validation",
            auto_id=True,
            shuffle=True,
        )

        # Build a mapping from question text to embeddings
        question_embedding_map = {
            s.input: torch.tensor(s.metadata["embedding"]) for s in ds
        }

    # Construct the solver list: the main adaptive solver plus (if specified) a judge solver
    solver_list = [
        adaptive_truthfulqa_solver_refactor(
            initial_log_path=initial_log_path,
            n_positive_samples=n_positive_samples,
            n_negative_samples=n_negative_samples,
            generator_model_name=generator_model_name,
            eval_model_name=eval_model_name,
            self_check_model_name=self_check_model_name,
            use_embeddings=use_embeddings,
            embeddings_model_name=embeddings_model_name,
            question_embedding_map=question_embedding_map,
            similarity_threshold=similarity_threshold,
            score_threshold=score_threshold,
            max_attempts=max_attempts,
            randomize_sampling=randomize_sampling,
            cot_in_context=cot_in_context,
            use_cot_generator=use_cot_generator,
            use_cot_evaluator=use_cot_evaluator,
            original_eval_model_name=original_eval_model_name,
            use_eval_model_for_checker=use_eval_model_for_checker,
        ),
    ]

    # if we have a separate judge model, append the judge solver
    if judge_model_name:
        solver_list.append(
            adaptive_truthfulqa_judge_solver(
                initial_log_path=initial_log_path,
                judge_model_name=judge_model_name,
            )
        )

    scorer_list = [
        adaptive_truthfulqa_scorer(),
    ]
    # if we have a judge model, also include judge-based scorers
    if judge_model_name:
        scorer_list.append(adaptive_truthfulqa_judge_scorer())
        scorer_list.append(adaptive_truthfulqa_scorer_judged())

    return Task(
        dataset=MemoryDataset(name="adaptive_truthfulqa_refactor", samples=[]),
        solver=solver_list,
        scorer=scorer_list,
        epochs=Epochs(
            epochs=5,
            reducer=[
                "mean",
                novelty_filter_judged_only(
                    embeddings_model_name=embeddings_model_name,
                    similarity_threshold=similarity_threshold,
                    scorer_id="adaptive_truthfulqa_scorer_judged"  # so we only apply the novelty check/compute the embeddings on the final judged scores
                )
            ]
        ),
    )

@task
def press_truthfulqa(
    initial_log_path: str,
    n_iterations: int = 5,
    quiz_length: int = 10,
    generator_model_name: str = "openai/gpt-4o",
    evaluator_model_name: str = "openai/gpt-4o-mini",
    eval_model_name: str = "openai/gpt-4o-mini",
    target: Literal["mc1", "mc2"] = "mc1",
) -> Task:
    """
    Adaptive TruthfulQA task that implements the PRESS Method to generate report cards
    and generates new questions based on these summaries.
    """
    return Task(
        dataset=MemoryDataset(name="press_truthfulqa", samples=[]),
        solver=[
            press_truthfulqa_solver(
                initial_log_path=initial_log_path,
                n_iterations=n_iterations,
                quiz_length=quiz_length,
                generator_model_name=generator_model_name,
                evaluator_model_name=evaluator_model_name,
                eval_model_name=eval_model_name,
                target=target,
            ),
        ],
        scorer=press_truthfulqa_scorer(),
    )


@task
def re_evaluate_adaptive_truthfulqa(
    adaptive_log_path: str,
    new_model_name: str = "openai/gpt-4o-mini",
    multiple_correct: bool = False,
    temperature: float = 0.0,
    use_cot: bool = False,
    custom_template: Optional[str] = None,
) -> Task:
    """
    Re-evaluates the newly generated adaptive TruthfulQA questions on a new model.

    Usage:
        inspect eval truthfulqa.py -T re_evaluate_adaptive_truthfulqa
            -T adaptive_log_path=PATH_TO_ADAPTIVE_LOG
            -T new_model_name="openai/gpt-4o-mini"
            -T multiple_correct=False
            -T temperature=0.0
            -T use_cot=False
            -T custom_template=None

    Args:
        adaptive_log_path (str): Path to the final log from the adaptive_truthfulqa_solver.
        new_model_name (str): The model name to evaluate on filtered, judge-accepted questions.
        multiple_correct (bool): Whether there may be multiple correct answers to each question.
        temperature (float): Model generation temperature.
        use_cot (bool): If True, use chain-of-thought prompting with SINGLE_ANSWER_TEMPLATE_COT.
        custom_template (Optional[str]): Custom template override. Must contain {question} and {choices} placeholders.

    Returns:
        Task object that, when run via "inspect eval", will load the log, filter
        judge-approved questions, have the solver prompt the new model, then
        compute how many the new model got right among those judge-approved questions.
    """
    return Task(
        # We keep an empty MemoryDataset since the real data comes from the saved log
        dataset=MemoryDataset(name="re_evaluate_adaptive_truthfulqa", samples=[]),
        solver=[
            evaluate_adaptive_truthfulqa_questions(
                adaptive_log_path=adaptive_log_path,
                new_model_name=new_model_name,
                multiple_correct=multiple_correct,
                temperature=temperature,
                use_cot=use_cot,
                custom_template=custom_template,
            )
        ],
        # We use a simple custom scorer that checks "re_evaluation_samples"
        scorer=[re_evaluation_scorer()],
        epochs=Epochs(
            epochs=1,
            reducer=["mean"],  # or other aggregator if desired
        ),
    )

if __name__ == "__main__":
    # model_list_generator = ["openai/gpt-4o", "openai/gpt-4o-mini", "together/mistralai/Mixtral-8x22B-Instruct-v0.1", "anthropic/claude-3-5-sonnet-20240620"]
    model_list_generator = ["together/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"]
    model_list_eval = ["openai/gpt-4o-mini"]
    # model_list_eval = ["openai/gpt-4o", "openai/gpt-4o-mini", "together/mistralai/Mixtral-8x22B-Instruct-v0.1", "anthropic/claude-3-5-sonnet-20240620"]

    for eval_model in model_list_eval:
        # first, run the initial truthfulqa task
        task = truthfulqa_initial(target="mc1")
        log_dir = f"new_logs/initial_truthfulqa_log_{eval_model.replace('/', '_')}"
        if os.path.exists(log_dir):
            # check if any json exists in log_dir
            json_files = [f for f in os.listdir(log_dir) if f.endswith('.json')]
        else:
            json_files = []

        if json_files:
            try:
                initial_log_path = os.path.join(log_dir, max(
                    json_files,
                    key=lambda x: os.path.getctime(os.path.join(log_dir, x))
                ))
                assert os.path.exists(initial_log_path), f"Initial log path {initial_log_path} does not exist"
                print(f"Skipping initial truthfulqa task for {eval_model} because it already exists, in {log_dir}, called {initial_log_path}")
                initial_log = read_eval_log_async(initial_log_path)
                assert initial_log.status == "success", f"Initial log {initial_log_path} did not complete successfully"
            except Exception as e:
                print(f"An error occurred while retrieving the latest JSON file: {e}")
                eval(task, epochs=Epochs(1, "max"), max_connections=10000, log_dir=log_dir, model=eval_model)[0]
                initial_log_path = os.path.join(log_dir, max(
                        [f for f in os.listdir(log_dir) if f.endswith('.json')],
                    key=lambda x: os.path.getctime(os.path.join(log_dir, x))
                ))
                assert os.path.exists(initial_log_path), f"Initial log path {initial_log_path} does not exist"
                initial_log = read_eval_log_async(initial_log_path)
                assert initial_log.status == "success", f"Initial log {initial_log_path} did not complete successfully"
        else:
            print(f"No JSON files found in {log_dir}. Proceeding with the initial truthfulqa task.")
            eval(task, epochs=Epochs(1, "max"), max_connections=10000, log_dir=log_dir, model=eval_model)[0]
            initial_log_path = os.path.join(log_dir, max(
                    [f for f in os.listdir(log_dir) if f.endswith('.json')],
                key=lambda x: os.path.getctime(os.path.join(log_dir, x))
            ))

        for generator_model in model_list_generator:
            # then, run the adaptive truthfulqa task
            for positive_samples in [1]:
                for negative_samples in [8]:
                    log_dir = f"new_logs/initial_adaptive_truthfulqa_log_{eval_model.replace('/', '_')}"
                    # get latest json in log_dir
                    print(f"initial_log_path: {initial_log_path}")
                    task = adaptive_truthfulqa(
                        initial_log_path=initial_log_path,
                        n_positive_samples=positive_samples,
                        n_negative_samples=negative_samples,
                        generator_model_name=generator_model,
                        eval_model_name=eval_model,
                        target="mc1",
                        use_cot=False,
                    )
                    eval(task, epochs=Epochs(30, "mean"), max_connections=10000, log_dir=log_dir, model=eval_model, temperature=0)[0] 

                    task = adaptive_truthfulqa(
                        initial_log_path=initial_log_path,
                        n_positive_samples=positive_samples,
                        n_negative_samples=negative_samples,
                        generator_model_name=generator_model,
                        eval_model_name=eval_model,
                        target="mc1",
                        use_cot=True,
                    )
                    eval(task, epochs=Epochs(30, "mean"), max_connections=10000, log_dir=log_dir, model=eval_model, temperature=0)[0] 