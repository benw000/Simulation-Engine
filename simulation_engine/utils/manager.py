import numpy as np
from pathlib import Path
import datetime
from simulation_engine.classes.parents import Particle, Environment, Wall, Target
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

class Manager:
    '''
    Manages the collection of particle and environment entities
    '''
    # =============================================================

    # ---- INITIALISATION ---- 

    # TODO: Clear up inputs
    def __init__(self,
                 delta_t:float = 0.01,
                 num_timesteps:int = 100,
                 store_history:bool=True,
                 save_json:bool = True,
                 json_path:Path = None,
                 save_mp4:bool = True,
                 mp4_path:Path = None,
                 compute_and_animate:bool = True,
                 show_graph:bool = False,
                 torus:bool = False,
                 track_com:bool = False):
        """
        Initialise a Manager object to oversee the timestepping and state storing
        Function called by a simulation mode's setup function
        """
        # TODO: Validate inputs
        if torus and track_com:
            raise NotImplementedError("Please pick one of torus or track_com")

        # Initialise state dictionary
        self.state = {"Particle":{},
                      "Environment":{}}
        
        # Initialise max IDs for particle
        self.max_ids_dict = {}

        # Initialise history
        self.store_history = store_history
        self.history = []

        # Time
        self.delta_t = delta_t
        self.num_timesteps = num_timesteps
        self.current_time = 0
        self.current_step = 0

        # Graph
        self.show_graph = show_graph

        # Logging
        self.compute_and_animate = compute_and_animate
        self.save_json = save_json
        self.save_mp4 = save_mp4
        self.json_path = json_path
        self.mp4_path = mp4_path
        if self.json_path is None and self.save_json:
            self.json_path = Path(f"simulation_log_{str(datetime.datetime.now())}.json")
        if self.mp4_path is None and self.save_mp4:
            self.json_path = Path(f"simulation_vid_{str(datetime.datetime.now())}.mp4")
        
    # =============================================================

    # ---- DUNDER ---- 
    def __iter__(self):
        for class_name, class_dict in self.state["Particle"].items():
            print("DEBUG")
            print(class_name)
            print(class_dict)
            for id, particle in class_dict.items():
                if particle.alive:
                    yield particle

    def iterate_all_particles(self):
        for class_name, class_dict in self.state["Particle"].items():
            for id, particle in class_dict.items():
                if particle.alive:
                    yield particle
    
    # =============================================================

    # ---- UPDATE ---- 
    def update(self):
        """
        To be called at each simulation timestep
        """
        # Update state
        for particle in self:
            particle.update()

        # Increment time
        self.current_time += self.delta_t
        self.current_step += 1

        # TODO: Update kill records

        # Store in history
        if self.store_history:
            self.history.append(self.state)

        # Write to file
        self.append_state_to_json()
    
    # =============================================================

    # ---- LOGGING ----
    def append_state_to_json(self):
        if not self.json_path.exists():
            # TODO: Initialise file with python touch equivalent
            # Can we append to JSON without having to read it? 
            # Could we store in SQL db?
            pass
    
    # ---- LOADING ----
    def load_timestep(self, timestep):
        # Try loading state from history, if fails try from JSON
        if self.store_history:
            self.state = self.history[timestep]
        # Load from JSON
        else:
            # TODO: Read from JSON when history not saved
            raise NotImplementedError("Reading state from JSON not implemented yet")

    # =============================================================
    
    # ---- MATPLOTLIB PLOTTING ----

    # ---- ANIMATION ----
    def animate_plt(self):
        # Set up figure
        fig, ax, ax2 = self.setup_figure_plt()

        # Setup a matplotlib FuncAnimation of draw_figure_plt
        # (draw_figure_plt handles updating/loading state)
        interval_between_frames = self.delta_t*1000 # milliseconds
        ani = FuncAnimation(fig, self.draw_figure_plt,
                            frames=self.num_timesteps,
                            fargs=([ax],[ax2]), 
                            interval=interval_between_frames)
        plt.show()

        # Optionally save video
        if self.save_mp4:
            fps = 1/self.delta_t # period -> frequency
            ani.save(self.mp4_path, writer='ffmpeg', fps=fps)
            print("\n")
            print(f"Saved simulation as mp4 at {self.mp4_path}.")

    # ---- SETUP FIGURE ----
    def setup_figure_plt(self):
        # Initialise as None
        fig, ax, ax2 = None, None, None

        if self.show_graph:
            # Setup figure with 2 subplots
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

            # Initialise ax2 by plotting dimensions
            max_time = int(self.num_timesteps)*self.delta_t
            ax2.set_xlim(0, max_time) 

        else:
            # Setup figure
            fig, ax = plt.subplots()#figsize=[20,15])
            fig.set_size_inches(20,20)

        # Initialise ax by plotting dimensions
        ax.set(xlim=[0,Particle.env_x_lim], ylim=[0,Particle.env_x_lim])
        
        # Add title
        window_title = "Simulation" # TODO: change this
        fig.canvas.set_window_title(window_title)

        # Set layout for ax
        ax.clear()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.set_aspect('equal', adjustable='datalim')
        fig.tight_layout()

        # Draw backdrop with either default or user supplied backdrop
        ax = self.draw_backdrop_plt_func(ax)

        # Initialise the matplotlib 'artists' for each particle and background sprite
        ax = self.setup_frame_plt(ax)
        return fig, ax, ax2
    
    def setup_frame_plt(self, ax):
        # Decide if tracking the COM in each frame
        com, scene_scale = None, None
        if Particle.track_com:
            com = Particle.centre_of_mass()
            scene_scale = Particle.scene_scale()

        # Update particle and environment artists
        for environment_object_type, environment_objects_list in self.state["Environment"].items():
            for environment_object in environment_objects_list:
                environment_object.init_plt(ax, com,scene_scale)
        for particle in self.iterate_all_particles():
            particle.init_plt(ax, com,scene_scale)

        return ax
    
    # ---- DRAW FIGURE ----
    def draw_figure_plt(self, timestep, ax, ax2=None):
        # Update or load state
        if self.compute_and_animate:
            self.update()
        else:
            self.load_timestep(timestep)

        # Unpack ax, ax2 from FuncAnimation's wrapper
        ax = ax[0]
        ax2 = ax2[0]

        # Print calculation progress
        print(f"----- Animation progress: {timestep} / {self.num_timesteps} -----" ,end="\r", flush=True)

        # Draw frame and graph onto existing axes
        self.draw_frame_plt(ax)
        if self.show_graph:
            self.draw_graph_plt(ax2)
    
    # ---- DRAW FRAME ----
    def draw_frame_plt(self, ax):
        '''
        Draw all objects in the simulation's current state onto a supplied matplotlib ax object.
        '''        
        # Decide if tracking the COM in each frame
        com, scene_scale = None, None
        if Particle.track_com:
            com = Particle.centre_of_mass()
            scene_scale = Particle.scene_scale()

        # Update particle and environment artists
        ax = self.draw_environment_objects_plt(ax, com, scene_scale)
        ax = self.draw_particle_objects_plt(ax, com, scene_scale)

        return ax

    @staticmethod
    def default_draw_backdrop_plt(ax):
        # Default background for matplotlib frames if not specified by user
        ax.set_xlim(-1, Particle.env_x_lim+1)
        ax.set_ylim(-1, Particle.env_y_lim+1)

        # Black background
        ax.set_facecolor('k')
        return ax
    
    draw_backdrop_plt_func = default_draw_backdrop_plt

    def draw_environment_objects_plt(self, ax, com, scene_scale):
        # Draw all environment objects from state dictionary onto supplied Matplotlib.pyplot ax object
        for environment_object_type, environment_objects_list in self.state["Environment"].items():
            for environment_object in environment_objects_list:
                environment_object.draw_plt(ax, com,scene_scale)
        return ax
    
    def draw_particle_objects_plt(self, ax, com, scene_scale):
        # Draw all particle objects from state dictionary onto supplied Matplotlib.pyplot ax object
        for particle in self.iterate_all_particles():
            particle.draw_plt(ax, com,scene_scale)
        return ax
    
    # ---- DRAW GRAPH ----
    def draw_graph_plt(self, ax2):
        # Oversee drawing a graph onto an axis
        # Add any shared functionality here
        self.draw_graph_plt_func(ax2)
        return ax2

    @staticmethod
    def default_draw_graph_plt(self, ax2):
        # Default graph if not specified by user
        max_time = int(self.num_timesteps)*self.delta_t
        ax2.set_xlim(0, max_time)  # Set x-axis limits
        return ax2
    
    draw_graph_plt_func = default_draw_graph_plt

    
