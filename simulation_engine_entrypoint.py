#!/usr/bin/env python3

import argparse
from pathlib import Path
from pathvalidate import validate_filepath
from pathvalidate.argparse import validate_filename_arg, validate_filepath_arg
import simulation_engine
from simulation_engine.classes import Manager

INTERACTIVE_SUPPORTED_TYPES = []

SETUP_FUNC_DICT = {
    "evac" : simulation_engine.classes.evac.setup,
    #"birds" : simulation_engine.classes.birds.setup,
    "nbody" : simulation_engine.classes.nbody.setup,
    #"springs": simulation_engine.classes.springs.setup,
    "pool": simulation_engine.classes.pool.setup
}

def validate_mode_args(args):
    # Unpack
    mode = args.mode
    type = args.type
    interactive = args.interactive

    # Can't run interactive when in load mode
    if mode=='load' and interactive:
        raise argparse.ArgumentError(f"(-i, --interactive): Cannot use interactive mode while in 'load' mode")

    # Can't run interactive for all modes
    if interactive and type not in INTERACTIVE_SUPPORTED_TYPES:
        raise NotImplementedError(f"Interactive mode not currently supported for {type} simulation type")


def validate_setup_args(args):
    # Number of time steps
    steps = args.steps
    if steps <= 0:
        raise argparse.ArgumentError(f"(-s, --steps): Please enter a positive (>0) integer number of time steps. Got {steps}")
    if steps > 1000:
        print(f"Warning: User has set high number of total time steps ({steps})")
    
    # Time step duration
    deltat = args.steps
    if deltat <= 0:
        raise argparse.ArgumentError(f"(-d, --deltat): Please enter a positive (>0) float number for timestep delta t duration (seconds). Got {deltat}")
    
    # Number of classes - note individual setup function needs to validate length of list
    nums = args.nums
    for num in nums:
        if num < 0:
            raise argparse.ArgumentError(f"(-n, --nums): Please enter positive (>0) integer number(s) for starting population(s) of particles. Got {nums}")
        if num > 1000:
            print(f"Warning: User has set high starting population(s) ({nums})")

def validate_memory_args(args):
    # Unpack
    sync = args.sync
    cache = args.cache
    log = args.log
    vid = args.vid

    # If not in sync mode, must store some sort of history
    if sync and not cache and not log:
        raise argparse.ArgumentError(f"If not running in sync mode (-s), then at least one of caching (-c, --cache) or logging (-l, --log) must be true. Otherwise can't store a history to draw back on.")

def validate_filepath_args(args):
    # Unpack
    log_name = args.log_name
    log_folder = args.log_folder
    log_path = args.log_path
    vid_name = args.vid_name
    vid_folder = args.vid_folder
    vid_path = args.vid_path

    # Log path combination
    if log_path and (log_name or log_folder):
        raise argparse.ArgumentError("Please only supply (--log_path) or (--log_name AND/OR --log_folder)")
    
    # Vid path combination
    if vid_path and (vid_name or vid_folder):
        raise argparse.ArgumentError("Please only supply (--vid_path) or (--vid_name AND/OR --vid_folder)")

    # We make use of the 'pathvalidate' package to check the supplied paths for valid syntax (not exist checks)
    # pathvalidate.argparse.[validate_filename_arg, validate_filepath_arg] is used in argparse
    # We just need to check the folder by setting up a dummy full path using it

    # Log folder
    if log_folder:
        dummy_path = Path(log_folder).absolute() / "dummy.ndjson"
        validate_filepath(dummy_path)
    
    # Vid folder
    if vid_folder:
        dummy_path = Path(vid_folder).absolute() / "dummy.mp4"
        validate_filepath(dummy_path)

def main(args):
    # ---- VALIDATE INPUTS ----
    validate_mode_args(args)
    validate_setup_args(args)
    validate_memory_args(args)
    validate_filepath_args(args)

    # Call setup with user inputs
    setup_func =  SETUP_FUNC_DICT[args.mode]
    manager: Manager = setup_func(num_time_steps, delta_t, nums, save_json, json_path, save_mp4, mp4_path, compute_and_animate)

    # TODO: Add splitting for matplotlib, pygame, etc
    if not compute_and_animate:
        print("Starting Computation")
        for t in range(num_time_steps):
            manager.update()
        print("Finished")
    
    
    # Animate when done
    manager.animate_plt()
    print("DONE!")

    # TODO: View final video embedded in window, or print done

if __name__=="__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="General Simulation Engine input options.")
    
    # ---- Arguments ----
    # Mode and type of simulation
    parser.add_argument('mode', type=str, choices=["run", "load"], help="The mode to run the simulation in: 'run' for normal simulation building, 'load' to load from a ndjson log", default="run")
    parser.add_argument('type', type=str, choices=["evac", "birds", "nbody", "springs", "pool"], help='The type of simulation to run')
    parser.add_argument('-i','--interactive', action='store_true', help="Use this flag to in run mode to run interactively (like a game)", default=False)
    # Simulation setup
    parser.add_argument('-t','--steps', type=int, help='The number of timesteps in the simulation', default=100)
    parser.add_argument('-d','--deltat', type=float, help='The duration of each timestep in seconds', default=0.01)
    parser.add_argument('-n','--nums', nargs='+', type=int, help='The number of particles in each class for a multi-class simulation. List of ints e.g 1 4 5', default=None)
    # Simulation memory and saving
    parser.add_argument('-s','--sync', action='store_true', help="Use this flag to synchronously animate each frame as soon as it's computed - otherwise simulation is fully computed and then animated", default=False)
    parser.add_argument('-c','--cache', type=bool, choices=[True,False], help='Whether to store/cache simulation history in memory', default=True)
    parser.add_argument('-l','--log', type=bool, choices=[True,False], help="Whether to write simulation history to an NDJSON log file", default=True)
    parser.add_argument('-v','--vid',action='store_true', help='Use this flag to save the simulation as an MP4 video', default=False)
    # Custom file paths
    parser.add_argument('--log_name', type=validate_filename_arg,   help="Custom file name (not path) for NDJSON log", default=None)
    parser.add_argument('--log_folder', type=str, help="Custom folder to store NDJSON log in", default=None)
    parser.add_argument('--log_path', type=validate_filepath_arg,   help="Custom file path for NDJSON log", default=None)
    parser.add_argument('--vid_name', type=validate_filename_arg,   help="Custom video name (not path) for MP4 video", default=None)
    parser.add_argument('--vid_folder', type=str, help="Custom folder to store MP4 video in", default=None)
    parser.add_argument('--vid_path', type=validate_filepath_arg,   help="Custom file path for MP4 video", default=None)

    # Parse the arguments
    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    main(args)