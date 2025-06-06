import unittest
from unittest.mock import patch
from unittest import main
from simulation_engine_entrypoint import IMPLEMENTED_TYPES
from simulation_engine.utils.errors import SimulationEngineInputError

class TestSimulationEngineEntrypoint(unittest.TestCase):
    entry_script = "simulation_engine_entrypoint.py"
    print_func = "simulation_engine_entrypoint.print"

    # ------------------------------------------------------
    # Bad input combinations

    def _generate_bad_input_combinations(self):
        """
        Generates bad input arguments for use in test_bad_input_combinations
        """
        # TODO
        raise NotImplementedError
        # eg ["--name", "Alice"]

    def test_bad_input_combinations(self):
        """
        Factory function to create test cases for each bad input argument
        """
        # Get bad inputs
        inputs_list = self._generate_bad_input_combinations()
        for args_list in inputs_list:
            # Create function
            @patch("sys.argv", [self.entry_script]+args_list)
            def test_missing_args(self):
                # Call main and collect errors
                # TODO: Allow argparse error or custom error
                with self.assertRaises(SystemExit) as cm:
                    main.main()
                self.assertEqual(cm.exception.code, 2)
    
    @patch("sys.argv", [self.entry_script]+args_list)
    def test_correct_input(self):
        # Redirect calls to rich print --> mock print object
        with patch(self.print_func) as mock_print:
            # Call main 
            main.main()


if __name__=="__main__":
    unittest.main()