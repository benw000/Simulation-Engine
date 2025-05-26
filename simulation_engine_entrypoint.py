#!/usr/bin/env python3
import json
import argparse
from rich import print
from pathlib import Path
from pathvalidate import validate_filepath
from pathvalidate.argparse import validate_filename_arg, validate_filepath_arg

import simulation_engine
from simulation_engine.classes import Manager
from simulation_engine.utils.errors import SimulationEngineInputError

# ---- TO BE MAINTAINED ----
# Map from string to module's setup
SETUP_FUNC_DICT = {
    "evac" : simulation_engine.classes.evac.setup,
    "birds" : simulation_engine.classes.birds.setup,
    "nbody" : simulation_engine.classes.nbody.setup,
    "springs": simulation_engine.classes.springs.setup,
    "pool": simulation_engine.classes.pool.setup
}
# Track implementations
IMPLEMENTED_TYPES = ["nbody"]
INTERACTIVE_SUPPORTED_TYPES = []

# ---- VALIDATION FUNCTIONS ----
def validate_mode_args(args):
    """
    Validate the mode, type, interactive arguments supplied by the user

    Args:
        args (argparse.Namespace): Namespace containing input arguments

    Raises:
        NotImplementedError: If a feature hasn't been implemented
        SimulationEngineInputError: If supplied arguments clash
    """
    # Unpack
    mode = args.mode
    type = args.type
    interactive = args.interactive

    # Can't fill in type unless in load mode
    if not type and not mode == 'load':
        raise SimulationEngineInputError(f"(argument 2): Please supply a simulation type from the list {IMPLEMENTED_TYPES}")

    # Can't run unimplemented type
    if type not in IMPLEMENTED_TYPES+[None]:
        raise NotImplementedError(f"Type {type} not yet implemented!")
    
    # Can't run interactive when in load mode
    if mode=='load' and interactive:
        raise SimulationEngineInputError(f"(-i, --interactive): Cannot use interactive mode while in 'load' mode")

    # Can't run interactive for all modes
    if interactive and type not in INTERACTIVE_SUPPORTED_TYPES:
        raise NotImplementedError(f"Interactive mode not currently supported for {type} simulation type")

def validate_setup_args(args):
    """
    Validate the steps, deltat, nums arguments supplied by the user

    Args:
        args (argparse.Namespace): Namespace containing input arguments

    Raises:
        NotImplementedError: If a feature hasn't been implemented
        SimulationEngineInputError: If supplied arguments clash
    """
    # Number of time steps
    steps = args.steps
    if steps <= 0:
        raise SimulationEngineInputError(f"(-s, --steps): Please enter a positive (>0) integer number of time steps. Got {steps}")
    if steps > 1000:
        print(f"Warning: User has set high number of total time steps ({steps})")
    
    # Time step duration
    deltat = args.steps
    if deltat <= 0:
        raise SimulationEngineInputError(f"(-d, --deltat): Please enter a positive (>0) float number for timestep delta t duration (seconds). Got {deltat}")
    
    # Number of classes - note individual setup function needs to validate length of list
    nums = args.nums
    if nums:
        for num in nums:
            if num < 0:
                raise SimulationEngineInputError(f"(-n, --nums): Please enter positive (>0) integer number(s) for starting population(s) of particles. Got {nums}")
            if num > 1000:
                print(f"Warning: User has set high starting population(s) ({nums})")

def validate_memory_args(args):
    """
    Validate the sync, cache, log, vid arguments supplied by the user

    Args:
        args (argparse.Namespace): Namespace containing input arguments

    Raises:
        NotImplementedError: If a feature hasn't been implemented
        SimulationEngineInputError: If supplied arguments clash
    """
    # Unpack
    sync = args.sync
    cache = args.cache
    log = args.log
    vid = args.vid

    # If not in sync mode, must store some sort of history
    if sync and not cache and not log:
        raise SimulationEngineInputError(f"If not running in sync mode (-s), then at least one of caching (-c, --cache) or logging (-l, --log) must be true. Otherwise can't store a history to draw back on.")

def validate_filepath_args(args):
    """
    Validate the log and video path arguments supplied by the user

    Args:
        args (argparse.Namespace): Namespace containing input arguments

    Raises:
        NotImplementedError: If a feature hasn't been implemented
        SimulationEngineInputError: If supplied arguments clash
    """
    '''
    We make use of the 'pathvalidate' package to check the supplied paths for valid syntax (not exist checks)
    We already check names and paths in the argparse below with types:
        pathvalidate.argparse.[validate_filename_arg, validate_filepath_arg]
    '''
    # Unpack
    mode = args.mode
    log_name = args.log_name
    log_folder = args.log_folder
    log_path = args.log_path
    chunk_size = args.log_read_chunk_size
    vid_name = args.vid_name
    vid_folder = args.vid_folder
    vid_path = args.vid_path

    # Log path combination
    if log_path and (log_name or log_folder):
        raise SimulationEngineInputError("Please only supply (--log_path) or (--log_name AND/OR --log_folder)")
    
    # Log path suffix
    if log_name:
        if not Path(log_name).suffix == ".ndjson":
            raise SimulationEngineInputError("If supplying (--log_name) please use a .ndjson suffix")

    # Log name suffix
    if log_path:
        if not Path(log_path).suffix == ".ndjson":
            raise SimulationEngineInputError("If supplying (--log_path) please use a .ndjson suffix")

    # Log folder
    if log_folder:
        dummy_path = Path(log_folder).absolute() / "dummy.ndjson"
        validate_filepath(dummy_path)

    # Log chunk size
    if chunk_size < 0:
        raise SimulationEngineInputError(f"(--log_read_chunk_size): Please enter a positive (>0) integer number for chunk size (number of timestep states loaded). Got {chunk_size}")

    # Vid path combination
    if vid_path and (vid_name or vid_folder):
        raise SimulationEngineInputError("Please only supply (--vid_path) or (--vid_name AND/OR --vid_folder)")

    # Vid path suffix
    if vid_name:
        if not Path(vid_name).suffix == ".ndjson":
            raise SimulationEngineInputError("If supplying (--vid_name) please use a .ndjson suffix")

    # Vid name suffix
    if vid_path:
        if not Path(vid_path).suffix == ".ndjson":
            raise SimulationEngineInputError("If supplying (--vid_path) please use a .ndjson suffix")

    # Vid folder
    if vid_folder:
        dummy_path = Path(vid_folder).absolute() / "dummy.mp4"
        validate_filepath(dummy_path)

    # Valid log path needed in load mode
    if mode == 'load':
        # Log path can't be constructed
        if not (log_path or (log_folder and log_name)):
            raise SimulationEngineInputError(f"If using 'load' mode then please supply either (--log_path) or (--log_folder AND --log_name)")
        # Construct log path
        if not log_path:
            log_path = Path(log_folder) / log_name
        # Check log exists
        if not Path(log_path).exists():
            raise SimulationEngineInputError(f"Supplied log path doesnt exist: {log_path}")

def get_type(args):
    # Early return for non-load modes
    mode = args.mode
    if mode != "load":
        return args.type
    
    # Get log path
    log_path = args.log_path
    log_name = args.log_name
    log_folder = args.log_folder
    if not log_path:
        log_path = Path(log_folder) / Path(log_name)

    # Read first line of log for type
    with open(log_path, 'r') as f:
        first_line = f.readline()
        line_dict = json.loads(first_line)

    return line_dict["type"]

# ---- MAIN ---
def main(args):
    """
    Main entrypoint function for Simulation Engine.
    Does the following:
    1. Validates user-supplied arguments in args
    2. Sets up a Manager instance to oversee running, 
       referencing a setup function in SETUP_FUNC_DICT
    3. Tells the Manager to start

    Args:
        args (argparse.Namespace): Namespace containing input arguments

    Raises:
        NotImplementedError: If a feature hasn't been implemented
        SimulationEngineInputError: If supplied arguments clash
    """
    # ---- VALIDATE INPUTS ----
    validate_mode_args(args)
    validate_setup_args(args)
    validate_memory_args(args)
    validate_filepath_args(args)

    # ---- SETUP MANAGER ----
    args.type = get_type(args)
    setup_func =  SETUP_FUNC_DICT[args.type]
    manager: Manager = setup_func(args)
    manager.print_settings_table()
    
    # ---- RUN MANAGER ----
    manager.start()

if __name__=="__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="General Simulation Engine input options.")
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # ---- Arguments ----
    # Mode and type of simulation
    parser.add_argument('mode', type=str, choices=["run", "load"], help="The mode to run the simulation in: 'run' for normal simulation building, 'load' to load from a ndjson log")
    parser.add_argument('type', type=str, nargs='?', choices=IMPLEMENTED_TYPES, help='The type of simulation to run', default=None)
    parser.add_argument('-i','--interactive', action='store_true', help="Use this flag to in run mode to run interactively (like a game)", default=False)
    # Simulation setup
    parser.add_argument('-t','--steps', type=int, help='The number of timesteps in the simulation', default=100)
    parser.add_argument('-d','--deltat', type=float, help='The duration of each timestep in seconds', default=0.01)
    parser.add_argument('-n','--nums', nargs='+', type=int, help='The number of particles in each class for a multi-class simulation. List of ints e.g 1 4 5', default=None)
    # Simulation memory and saving
    parser.add_argument('-s','--sync', action='store_true', help="Use this flag to synchronously animate each frame as soon as it's computed - otherwise simulation is fully computed and then animated", default=False)
    parser.add_argument('-c','--cache', type=str2bool, help='Whether to store/cache simulation history in memory', default=True)
    parser.add_argument('-l','--log', type=str2bool, help="Whether to write simulation history to an NDJSON log file", default=True)
    parser.add_argument('-v','--vid',action='store_true', help='Use this flag to save the simulation as an MP4 video', default=False)
    # Custom file paths
    parser.add_argument('--log_name', type=validate_filename_arg,   help="Custom file name (not path) for NDJSON log", default=None)
    parser.add_argument('--log_folder', type=str, help="Custom folder to store NDJSON log in", default=None)
    parser.add_argument('--log_path', type=validate_filepath_arg,   help="Custom file path for NDJSON log", default=None)
    parser.add_argument('--log_read_chunk_size', type=int, help="Chunk size (number of timesteps loaded) for reading from log file", default=100)
    parser.add_argument('--vid_name', type=validate_filename_arg,   help="Custom video name (not path) for MP4 video", default=None)
    parser.add_argument('--vid_folder', type=str, help="Custom folder to store MP4 video in", default=None)
    parser.add_argument('--vid_path', type=validate_filepath_arg,   help="Custom file path for MP4 video", default=None)

    # Parse the arguments
    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    main(args)