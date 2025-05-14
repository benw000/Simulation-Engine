import argparse
import simulation_engine
import simulation_engine.classes

def main(args):
    # Unpack args
    mode = args.mode
    num_time_steps = args.steps
    delta_t = args.deltat
    nums = args.nums
    save_as_csv = args.save_csv
    user_csv_path = args.csv_path
    save_as_mp4 = args.save_mp4
    user_mp4_path = args.mp4_path

    # Setup based on mode
    setup_func_dict = {
        "evac" : simulation_engine.classes.evac.setup,
        "birds" : simulation_engine.classes.birds.setup,
        "nbody" : simulation_engine.classes.springs.setup,
        "springs": simulation_engine.classes.springs.setup,
        "pool": simulation_engine.classes.pool.setup
    }
    # Call setup with user inputs
    setup_func =  setup_func_dict[mode]
    manager = setup_func(num_time_steps, delta_t, nums, save_as_csv, save_as_mp4, user_csv_path, user_mp4_path)

    # Main timestepping
    for t in num_time_steps:
        manager.update()

    # TODO: View final video embedded in window, or print done





if __name__=="__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="General Simulation Engine input options.")
    
    # Add arguments
    parser.add_argument('-m','--mode', type=str, choices=["evac", "birds", "nbody", "springs", "pool"], help='The type of simulation to run')
    parser.add_argument('-s','--steps', type=int, help='The number of timesteps in the simulation [10 <= N <~ 500, default 100]', default=100)
    parser.add_argument('-d','--deltat', type=float, help='The duration of each timestep in seconds, default 0.01s', default=0.01)
    parser.add_argument('-n','--nums', nargs='+', type=int, help='The number of particles in each class for a multi-class simulation. List of ints e.g 1 4 5', default=None)
    parser.add_argument('-c','--save_csv', action='store_true', help='Use this flag to save the simulation timesteps in a compressed CSV log')
    parser.add_argument('--csv_path', type=str, help='Supply a custom path to the saved MP4', default=None)
    parser.add_argument('-m','--save_mp4', action="store_true", help="Use this flag to save the video as an MP4", default=False)
    parser.add_argument('--mp4_path', type=str, help='Supply a custom path to the saved MP4', default=None)

    # Parse the arguments
    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    main(args)