import unittest
from unittest.mock import patch

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import simulation_engine.main.entrypoint 
from simulation_engine.main.entrypoint import (
    IMPLEMENTED_TYPES, INTERACTIVE_SUPPORTED_TYPES, TYPE_DEFAULT_NUMS)
from simulation_engine.main.entrypoint import main as entrypoint_main

class TestInputArgs(unittest.TestCase):
    ENTRY_SCRIPT = simulation_engine.main.entrypoint.__file__

    # Common error types reference
    ERROR_CODES_LOOKUP = {
        "Success":0,
        "General Error":1,
        "Bad shell args":2,
    }

    # ------------------------------------------------------
    # Bad input combinations

    def _generate_bad_input_combinations(self):
        """
        Generates bad input arguments for use in test_bad_input_combinations
        """
        bad_path = "./?`§\|...\<>"
        # Create list of lists
        bad_inputs_list = [
            ["run"],
            ["load"],
            ["load", "-i"],
            ["run", IMPLEMENTED_TYPES[0], "-t", "-1"],
            ["run", IMPLEMENTED_TYPES[0], "-d", "-0.1"],
            ["run", IMPLEMENTED_TYPES[0], "-n", "0.5"],
            ["run", IMPLEMENTED_TYPES[0], "-n", "-1"],
            ["run", IMPLEMENTED_TYPES[0], "-n", "10", "-c", "false", "-l", "false"],
            ["run", IMPLEMENTED_TYPES[0], "-n", "10", "--log_path", "foo/bar.json", "--log_name", "bar.ndjson"],
            ["run", IMPLEMENTED_TYPES[0], "-n", "10", "--log_path", "foo/bar.json", "--log_folder", "foo"],
            ["run", IMPLEMENTED_TYPES[0], "-n", "10", "--log_path", bad_path],
            ["run", IMPLEMENTED_TYPES[0], "-n", "10", "--log_name", bad_path],
            ["run", IMPLEMENTED_TYPES[0], "-n", "10", "--log_folder", bad_path],
            ["run", IMPLEMENTED_TYPES[0], "-n", "10", "--vid_path", "foo/bar.mp4", "--vid_name", "bar.mp4"],
            ["run", IMPLEMENTED_TYPES[0], "-n", "10", "--vid_path", "foo/bar.mp4", "--vid_folder", "foo"],
            ["run", IMPLEMENTED_TYPES[0], "-n", "10", "--vid_path", bad_path],
            ["run", IMPLEMENTED_TYPES[0], "-n", "10", "--vid_name", bad_path],
            ["run", IMPLEMENTED_TYPES[0], "-n", "10", "--vid_folder", bad_path],
        ]
        # Run without number
        bad_inputs_list += [["run", sim_type] for sim_type in IMPLEMENTED_TYPES]

        # Load without log_path
        bad_inputs_list += [["load", sim_type] for sim_type in IMPLEMENTED_TYPES]

        # Unsupported interactive type
        bad_inputs_list += [["run", sim_type] for sim_type in IMPLEMENTED_TYPES if sim_type not in INTERACTIVE_SUPPORTED_TYPES]


        return bad_inputs_list

    def test_bad_input_combinations(self):
        """
        Function to test each set of bad input arguments
        """
        # Get bad inputs
        bad_inputs_list = self._generate_bad_input_combinations()
        for args_list in bad_inputs_list:
            print("Checking arguments:", args_list)
            # Call main with bad inputs
            with patch("sys.argv", [self.ENTRY_SCRIPT]+args_list):
                try:
                    entrypoint_main()
                # Check that argparse errors are correctly thrown
                except SystemExit as e:
                    code = e.code
                    print(f"Caught SystemExit with code {code}")
                    self.assertNotEqual(code, self.ERROR_CODES_LOOKUP["Success"],
                                        msg=f"Invalid input {args_list} exited with 0")
                # Else check that custom errors are correctly thrown
                except Exception as e:
                    print(f"Caught Exception: {e}")
                    continue
                # Fail test if no error thrown
                else:
                    self.fail(f"Invalid input {args_list} did not raise an error")

if __name__=="__main__":
    unittest.main()