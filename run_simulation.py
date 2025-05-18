import argparse
import simulation_engine
from simulation_engine.classes import Manager

SETUP_FUNC_DICT = {
    "evac" : simulation_engine.classes.evac.setup,
    #"birds" : simulation_engine.classes.birds.setup,
    "nbody" : simulation_engine.classes.nbody.setup,
    #"springs": simulation_engine.classes.springs.setup,
    "pool": simulation_engine.classes.pool.setup
}

def main(args):
    # TODO: Do a kargs thing to save doing this and pass all to function
    # Unpack args
    mode = args.mode
    num_time_steps = args.steps
    delta_t = args.deltat
    nums = args.nums
    save_as_csv = args.save_csv
    user_csv_path = args.csv_path
    save_as_mp4 = args.save_mp4
    user_mp4_path = args.mp4_path
    compute_and_animate = args.compute_and_animate

    # TODO: Validate inputs

    # Call setup with user inputs
    setup_func =  SETUP_FUNC_DICT[mode]
    manager: Manager = setup_func(num_time_steps, delta_t, nums, save_as_csv, save_as_mp4, user_csv_path, user_mp4_path, compute_and_animate)

    # TODO: Add splitting for matplotlib, pygame, etc
    if not compute_and_animate:
        for t in num_time_steps:
            manager.update()

    # Animate when done
    manager.animate_plt()

    # TODO: View final video embedded in window, or print done

if __name__=="__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="General Simulation Engine input options.")
    
    # Add arguments
    parser.add_argument('-m','--mode', type=str, choices=["evac", "birds", "nbody", "springs", "pool"], help='The type of simulation to run')
    parser.add_argument('-s','--steps', type=int, help='The number of timesteps in the simulation [10 <= N <~ 500, default 100]', default=100)
    parser.add_argument('-d','--deltat', type=float, help='The duration of each timestep in seconds, default 0.01s', default=0.01)
    parser.add_argument('-n','--nums', nargs='+', type=int, help='The number of particles in each class for a multi-class simulation. List of ints e.g 1 4 5', default=None)
    parser.add_argument('-c','--save_csv', action='store_true', help='Use this flag to save the simulation timesteps in a compressed CSV log', default=False)
    parser.add_argument('--csv_path', type=str, help='Supply a custom path to the saved MP4', default=None)
    parser.add_argument('--save_mp4', action="store_true", help="Use this flag to save the video as an MP4", default=False)
    parser.add_argument('--mp4_path', type=str, help='Supply a custom path to the saved MP4', default=None)
    parser.add_argument('--compute_and_animate', action='store_true', help='Use this flag to animate each frame as its computed, otherwise it will be stored to a history and animated when done', default=False)

    # Parse the arguments
    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    main(args)