import unittest
import subprocess
import shutil

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import simulation_engine.main.entrypoint 
from simulation_engine.main.entrypoint import (
    IMPLEMENTED_TYPES, INTERACTIVE_SUPPORTED_TYPES, TYPE_DEFAULT_NUMS)

class TestIntegration(unittest.TestCase):
    # Path to entrypoint
    ENTRY_SCRIPT = simulation_engine.main.entrypoint.__file__

    # Folders
    LOGS_FOLDER = "tests/data/Simulation_Logs/"
    VIDEOS_FOLDER = "tests/data/Simulation_Videos/"

    # Common error types reference
    ERROR_CODES_LOOKUP = {
        "Success":0,
        "General Error":1,
        "Bad shell args":2,
    }

    def base_run_args(self, sim_type):
            # Get base args to run in headless mode
            return ["run", sim_type, 
                    "--display", "False",
                    "--log_folder", self.LOGS_FOLDER,
                    "--vid_folder", self.VIDEOS_FOLDER,
                    "-n", ]+TYPE_DEFAULT_NUMS[sim_type]
    
    def _generate_quick_list(self):
        # Initialise store
        inputs_list = []
        for sim_type in IMPLEMENTED_TYPES:
            base = self.base_run_args(sim_type)
            inputs_list += [
                base,
                base+["-c", "False"],
                base+["-s"],
                base+["-v"],
            ]
            # For each type create a log, then load
            log_path = f"tests/data/Simulation_Logs/{sim_type}_load_test.ndjson"
            inputs_list += [
                ["run", sim_type, 
                "--display", "False",
                "--log_path", log_path,
                "-n", ]+TYPE_DEFAULT_NUMS[sim_type],
                ["load", 
                 "--display", "False",
                 "--log_path", log_path]
            ]
        return inputs_list

    def _generate_exhaustive_list(self):
        # Initialise store
        inputs_list = []

        # Loop over main logging/caching/synchronous modes
        for sim_type in IMPLEMENTED_TYPES:
            base = self.base_run_args(sim_type)
            inputs_list += [
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
            inputs_list += [
                ["run", sim_type, 
                "--display", "False",
                "--log_path", log_path,
                "-n", ]+TYPE_DEFAULT_NUMS[sim_type],
                ["load", 
                 "--display", "False",
                 "--log_path", log_path]
            ]
            
        # For each current one try rendering a video
        for args_list in inputs_list.copy():
            inputs_list.append(args_list+["-v"])
        
        return inputs_list
        
    def test_integration(self):
        """
        Integration test to try each valid CLI input.
        """
        # Empty old test output
        shutil.rmtree(self.LOGS_FOLDER)
        shutil.rmtree(self.VIDEOS_FOLDER)

        # Get list of input argument sets
        exhaustive = False
        if exhaustive:
            inputs_list = self._generate_exhaustive_list()
        else:
            inputs_list = self._generate_quick_list()

        # Test each input in list
        for i, args_list in enumerate(inputs_list):
            print("----"*5 + f"\n[{i+1}/{len(inputs_list)}] Checking arguments:\n", args_list, "\n")
            # Call main with good inputs
            try:
                subprocess.run(['python', self.ENTRY_SCRIPT]+args_list)
            except subprocess.CalledProcessError as e:
                raise e
            # Reset matplotlib
            print("Above arguments worked!\n")
       
if __name__=="__main__":
    unittest.main()

'''
TODO
Get pyproject setup with different dev installs
Update for interactive
'''