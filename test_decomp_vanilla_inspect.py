import unittest
from decomp_vanilla_inspect import gpqa_decomposition_task
from inspect_ai.solver._task_state import Sample
from inspect_ai import Epochs, eval

class TestDecompVanillaInspect(unittest.TestCase):
    """Unit tests for gpqa_decomposition_task in decomp_vanilla_inspect.py"""

    def test_json_parsing(self):
        """Test JSON parsing functionality in decomposition_solver_mcq"""

        # Sample JSON input similar to what the decomposition model might return
        json_input = '''
        {
            "step_1": "Analyze the question and identify key components.",
            "step_2": "Consider each answer choice individually.",
            "step_3": "Eliminate incorrect options based on analysis."
        }
        '''

        # Expected extraction result
        expected_output = {
            "step_1": "Analyze the question and identify key components.",
            "step_2": "Consider each answer choice individually.",
            "step_3": "Eliminate incorrect options based on analysis."
        }

        # Simulate the extract_json_decomp function
        from utils_plotting.common import extract_json_decomp
        extracted_data = extract_json_decomp(json_input, max_decompositions=3)

        self.assertEqual(extracted_data, expected_output)

    def test_end_to_end_run(self):
        """Test a simplified end-to-end run with a basic dataset"""

        # Create a basic sample
        sample_dataset = [
            Sample(
                input="What is the capital of France?",
                choices=["Paris", "London", "Rome", "Berlin"],
                target="A",
                id="1"
            )
        ]

        # Run the task with the sample dataset
        try:
            task = gpqa_decomposition_task(
                cot=True,
                epochs=1,
                max_iterations=1,
                max_decompositions=1,
            )
            # Override the task's dataset with the sample_dataset
            task.dataset = sample_dataset

            eval(
                task,
                epochs=Epochs(1, "max"),
                max_connections=10000,
                log_dir="test_decomp_vanilla_inspect",
                model="openai/gpt-3.5-turbo",
            )

            # If no exceptions occur, the test passes
            self.assertTrue(True)

        except Exception as e:
            self.fail(f"End-to-end run failed with exception: {e}")

if __name__ == "__main__":
    unittest.main() 