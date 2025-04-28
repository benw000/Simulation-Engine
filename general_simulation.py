import datetime
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from simulation_classes import *

import warnings
warnings.filterwarnings("ignore")

def main(args):
     # Unpack user arguments
    type = args.type
    time_steps = args.steps
    num = args.num
    num2 = args.num2
    save_as_mp4 = args.save_mp4
    user_mp4_path = args.mp4_path

    now = datetime.datetime.now()
    show_graph = False # secondary axis

    Particle.delta_t = 0.01

    # --------------------------------------------------------------------------------------------------------
    # Setup according to user-specified type

    if type == 'birds':
        Environment.background_type = "sky"
        num_prey = num
        num_pred = num2
        for i in range(num_prey):
            Prey()
        for i in range(num_pred):
            Predator()
        Particle.track_com = False
        Particle.torus = True
        csv_path = f"Simulation_CSVs/{type}_{str(num_prey)}_{str(num_pred)}_{str(now.time())}_{str(now.date())}.csv"
        mp4_path = f"Simulation_mp4s/{type}_{str(num_prey)}_{str(num_pred)}_{str(now.time())}_{str(now.date())}.MP4"
        window_title = f'Predator Prey animation, {num_prey} prey, {num_pred} predators'

    elif type == 'nbody':
        Environment.background_type = "space"
        Particle.walls_x_lim = 1000
        Particle.walls_y_lim = 1000
        specific = False
        if specific:
            Star(np.array([300,300]),np.array([200,100]))
            Star(np.array([600,500]),np.array([-100,300]))
            Star(np.array([400,700]),np.array([0,-100]))
        else:
            for i in range(num):
                Star()
        Particle.track_com = False
        Particle.torus = False
        csv_path = f"Simulation_CSVs/{type}_{str(num)}_{str(now.time())}_{str(now.date())}.csv"
        mp4_path = f"Simulation_mp4s/{type}_{str(num)}_{str(now.time())}_{str(now.date())}.MP4"
        window_title = f'N-body animation, {num} bodies'

    elif type == 'evac':
        Environment.background_type = "room"
        Particle.num_evacuees = num
        Particle.walls_x_lim = 10
        Particle.walls_y_lim = 10
        classroom = True
        first = False
        if classroom:
            Particle.walls_x_lim = 12
            Particle.walls_y_lim = 10
            x = Particle.walls_x_lim
            y = Particle.walls_y_lim

            wall_points = [[[0,0],[0,y]], # left wall
                           [[0,0],[x-0,0]], # bottom wall
                           [[0,y],[x-0,y]], # top wall
                           [[x-2,3.5],[x-2,6.5]], # big desk
                           [[x-0,0],[x,2]], # right wall bottom 
                           [[x,3],[x,7]], # right wall middle
                           [[x,8],[x-0,y]], # right wall top
                           #[[3,2],[3,4]],
                           #[[3,6],[3,8]],
                           [[4,2],[4,4]],
                           [[4,6],[4,8]],
                           #[[7,2],[7,4]],
                           #[[7,6],[7,8]],
                           [[7,2],[7,4]],
                           [[7,6],[7,8]]]
            
            for pair in wall_points:
                Wall(np.array(pair[0]), np.array(pair[1]))

            # Targets for each door
            Target(np.array([Particle.walls_x_lim+1,2.5]))
            Target(np.array([Particle.walls_x_lim+1,7.5]))

        elif first:
            Wall(np.array([0,0]),np.array([0,Particle.walls_y_lim]))
            Wall(np.array([0,0]),np.array([Particle.walls_x_lim, 0]))
            Wall(np.array([0,Particle.walls_y_lim]),np.array([Particle.walls_x_lim, Particle.walls_y_lim]))
            Wall(np.array([Particle.walls_x_lim, 0]),np.array([Particle.walls_x_lim-1, 4.5]))
            Wall(np.array([Particle.walls_x_lim-1, 5.5]),np.array([Particle.walls_x_lim, Particle.walls_y_lim]))
            Wall(np.array([3,5]),np.array([8, 5]))

        show_graph = True
        for i in range(num):
            Human()
        Particle.track_com = False
        Particle.torus = False
        csv_path = f"Simulation_CSVs/{type}_{str(num)}_{str(now.time())}_{str(now.date())}.csv"
        mp4_path = f"Simulation_mp4s/{type}_{str(num)}_{str(now.time())}_{str(now.date())}.MP4"
        window_title = f'Evacuation simulation, {num} people'
    
    elif type == "springs":
        Environment.background_type = "room"
        Particle.delta_t = 0.05
        Particle.walls_x_lim = 160
        Particle.walls_y_lim = 40
        for i in range(num):
            Solid()
        Particle.track_com = True
        csv_path = f"Simulation_CSVs/{type}_{str(num)}_{str(now.time())}_{str(now.date())}.csv"
        mp4_path = f"Simulation_mp4s/{type}_{str(num)}_{str(now.time())}_{str(now.date())}.MP4"
        window_title = f'Lattice of {num} solids connected by springs'

    elif type == "pool":
        csv_path = f"Simulation_CSVs/{type}_{str(now.time())}_{str(now.date())}.csv"
        mp4_path = f"Simulation_mp4s/{type}_{str(now.time())}_{str(now.date())}.MP4"
        window_title = f'Pool table breaking simulation'
        Particle.track_com = False
        Particle.torus = False
        Particle.delta_t = 0.01
        Pool.pool_setup()

    # --------------------------------------------------------------------------------------------------------
    # Create CSV file name

    csv_path = csv_path.replace(":","-") # makes file more readable
    Particle.csv_path = csv_path

    # --------------------------------------------------------------------------------------------------------
    # Loop through timesteps

    Particle.num_timesteps = time_steps
    
    
    for t in range(time_steps):
        # Print calculation progress
        print(f"----- Computation progress: {t} / {time_steps} -----" ,end="\r", flush=True)

        # Update system
        Particle.timestep_update()
    
    # --------------------------------------------------------------------------------------------------------
    # Animate the CSV

    print("-")
    print("\n")

    # Initialise a scatter plot (need all of this)
    if show_graph:
        #fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig = plt.figure(figsize=(10, 5))

        # Define a GridSpec layout to control the ratio between ax1 and ax2
        gs = GridSpec(1, 2, width_ratios=[1.5, 1])  # Both subplots will have equal width

        # Create subplots ax1 and ax2 using the GridSpec layout
        ax = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        # Set the aspect ratio for both ax1 and ax2
        ax.set_aspect(aspect=1.5)  # Set ax1 aspect ratio 15:10
        ax2.set_aspect(aspect=1.5)  # Set ax2 to match ax1

        # Adjust spacing between plots if necessary
        plt.subplots_adjust(wspace=0.3)
        scat = ax2.scatter([], [])


    else:
        fig, ax = plt.subplots()#figsize=[20,15])
        inverse_aspect_ratio = Particle.walls_y_lim/Particle.walls_x_lim
        window_x_size = 20
        #fig.set_size_inches(window_x_size,window_x_size*inverse_aspect_ratio)
        fig.set_size_inches(20,20)
    fig.canvas.set_window_title(window_title)
    fig.tight_layout()
    ax.set_xlim(-1, Particle.walls_x_lim+1)  # Set x-axis limits
    ax.set_ylim(-1, Particle.walls_y_lim+1)  # Set y-axis limits
    scat = ax.scatter([], [])
    


    # Animate frames by calling update() function
    interval_between_frames = Particle.delta_t*1000 # milliseconds
    if show_graph:
        ani = FuncAnimation(fig, Particle.animate_timestep, frames=time_steps, \
                        fargs=([ax],[ax2]), interval=interval_between_frames)
    else:
        ani = FuncAnimation(fig, Particle.animate_timestep, frames=time_steps, \
                        fargs=([ax],), interval=interval_between_frames)

    save_as_mp4 = True
    if save_as_mp4:
        mp4_path = mp4_path.replace(":","-")
        if user_mp4_path is not None:
            mp4_path = user_mp4_path
        fps = 1/(interval_between_frames*(10**(-3))) # period -> frequency
        ani.save(mp4_path, writer='ffmpeg', fps=fps)
        print("\n")
        print(f"Saved simulation as mp4 at {mp4_path}.")

    #plt.show()


    # Generate list of instances for each desired class
    # Compute datetime and file names
    # Update Particle.num_timesteps
    # for timestep in range(num_steps):
    #       nice print statement progress
    #       current_time = timestep*Particle.delta_t
    #       Particle.timestep_update()
    #       Particle.write_to_csv(filename)
    # print computing done, starting animation
    # fig, ax = plt.figure
    # ani = FuncAnimation( Particle.animation_timestep, fargs=(ax)   )


if __name__=="__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="General Simulation Engine input options.")
    
    # Add arguments
    parser.add_argument('--type', type=str, help='The type of simulation [evac, birds, nbody, springs, pool]')
    parser.add_argument('--steps', type=int, help='The number of timesteps in the simulation [10 <= N <~ 500, default 100]', default=100)
    parser.add_argument('--num', type=int, help='The number of particles in the simulation [1<= N <~ 500, default 20]', default=20)
    parser.add_argument('--num2', type=int, help='The number of secondary particles in the simulation (eg Predators) [1<= N <~ 500, default 3]', default=3)
    parser.add_argument('--save_mp4', type=bool, help='Whether to save the simulation as an mp4 video [True, False, default True]', default=True)
    parser.add_argument('--mp4_path', type=str, help="(Optional) The mp4's relative path string (will create within current directory).", default=None)

    # Parse the arguments
    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    main(args)
    