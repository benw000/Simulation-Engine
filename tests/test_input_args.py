import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import unittest
from unittest.mock import patch
import matplotlib.pyplot as plt

from simulation_engine_entrypoint import IMPLEMENTED_TYPES, INTERACTIVE_SUPPORTED_TYPES
from simulation_engine_entrypoint import main as entrypoint_main

class TestInputArgs(unittest.TestCase):
    entry_script = "simulation_engine_entrypoint.py"
    print_func = "simulation_engine_entrypoint.print"

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
            with patch("sys.argv", [self.entry_script]+args_list):
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

    # ------------------------------------------------------
    # Good input combinations

    TYPE_DEFAULT_NUMS = {
        "nbody":["10"],
        "birds":["10", "2"],
        "springs":["200"],
        "pool":["10"],
        "evac":["30"],
    }

    def _generate_good_input_combinations(self):
        # Get base args to run in headless mode
        def base_run_args(sim_type):
            return ["run", sim_type, 
                    "--display", "False",
                    "--log_folder", "tests/data/Simulation_Logs/",
                    "--vid_folder", "tests/data/Simulation_Videos",
                    "-n", ]+self.TYPE_DEFAULT_NUMS[sim_type]
        
        # Initialise store
        good_inputs_list = []

        # Loop over main logging/caching/synchronous modes
        for sim_type in IMPLEMENTED_TYPES:
            base = base_run_args(sim_type)
            good_inputs_list += [
                base,
                base+["-s"],
                base+["-c","False"],
                base+["-l","False"],
                base+["-s", "-c", "False"],
                base+["-s", "-l", "False"],
                base+["-s", "-c", "False", "-l", "False"]
            ]
            # For each type create a log, then load
            log_path = f"tests/data/Simulation_Logs/{sim_type}_load_test.ndjson"
            good_inputs_list += [
                ["run", sim_type, 
                "--display", "False",
                "--log_path", log_path,
                "-n", ]+self.TYPE_DEFAULT_NUMS[sim_type],
                ["load", 
                 "--display", "False",
                 "--log_path", log_path]
            ]
            
        # For each current one try rendering a video
        for args_list in good_inputs_list.copy():
            good_inputs_list.append(args_list+["-v"])
        
        return good_inputs_list
    
    def test_good_inputs(self):
        """
        Function to integration test each valid CLI input
        """
        # Get good inputs
        good_inputs_list = self._generate_good_input_combinations()
        for args_list in good_inputs_list:
            print("Checking arguments:", args_list)
            # Call main with bad inputs
            with patch("sys.argv", [self.entry_script]+args_list):
                entrypoint_main()
            # Reset matplotlib
            plt.close('all')
            print("Above arguments worked!\n")
    
        
if __name__=="__main__":
    unittest.main()

'''
TODO: Update for interactive
'''