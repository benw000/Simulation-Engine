import os
import sys
import cv2
import json
import shutil
import argparse
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime

from rich import print
from rich.table import Table
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TaskProgressColumn

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation

# Check if ffmpeg is available on PATH, and tell matplotlib to use it
ffmpeg_path = shutil.which("ffmpeg")
if not ffmpeg_path:
    sys.exit("Error: ffmpeg not found on system PATH. Please install ffmpeg and try again.")
import matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = ffmpeg_path

# Hide user warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from simulation_engine.classes.parents import Particle, Environment

class Manager:
    '''
    Manager class to oversee the main simulation engine pipeline.
    Holds a state dictionary of all Particle and Environment child classes,
    as well as all engine settings.
    '''
    # ---- INITIALISATION ---- 
    def __init__(self,
                 args:argparse.Namespace,
                 show_graph:bool = False,
                 draw_backdrop_plt_func = None,
                 draw_graph_plt_func = None,
                 ):
        """
        Initialise a Manager object to oversee the timestepping and state storing
        This will be called by a simulation type's setup() function,
        but is only completed after initialisation at the end of the setup() function.
        (After this, the main entrypoint script will call the Manager.start() method.)
        """
        # ---- Initialise stores ----
        # Initialise state dictionary
        self.state = {"Particle":{},
                      "Environment":{}}
        
        # Initialise max IDs for particle
        self.max_ids_dict = {}

        # Initialise child class registry (e.g. "Prey":Prey)
        self.class_objects_registry = {}

        # History of state dicts
        self.history = []

        # Make other classes point to this specific self instance
        Environment.manager = self
        Particle.manager = self

        # ---- Unpack entrypoint args ----
        # Mode arguments
        self.mode = args.mode
        self.simulation_type = args.type
        self.interactive = args.interactive
        # TODO: Decide on render framework to use from these
        self.render_framework = "matplotlib"

        # Time
        self.num_steps: int = args.steps
        if args.deltat is None:
            self.delta_t = 0.01
        else:
            self.delta_t: float = args.deltat
        self.current_time = 0
        self.current_step = 0
        self.done_rendering = False
        self.just_looped = False

        # Cache, log, sync bool flags
        self.write_log = args.log
        self.cache_history = args.cache
        self.sync_compute_and_rendering = args.sync
        self.save_video = args.vid

        # Construct log path from potentially incomplete input
        now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        default_log_name = Path(f"{args.type}_simulation_log_{now}.ndjson")
        default_log_folder = Path("./data/Simulation_Logs")
        default_log_path = default_log_folder / default_log_name
        if args.log_path:
            log_path = Path(args.log_path)
        elif args.log_name and args.log_folder:
            log_path = Path(args.log_folder) / Path(args.log_name)
        elif args.log_name:
            log_path = default_log_folder / Path(args.log_name)
        elif args.log_folder:
            log_path = Path(args.log_folder) / default_log_name
        else:
            log_path = default_log_path

        # Initialise Logger
        self.logger = Logger(manager=self, log_path=log_path, chunk_size=args.log_read_chunk_size)

        # Construct vid path from potentially incomplete input
        default_vid_name = Path(f"{args.type}_simulation_video_{now}.mp4")
        default_vid_folder = Path("./data/Simulation_Videos")
        default_vid_path = default_vid_folder / default_vid_name
        if args.vid_path:
            vid_path = Path(args.vid_path)
        elif args.vid_name and args.vid_folder:
            vid_path = Path(args.vid_folder) / Path(args.vid_name)
        elif args.vid_name:
            vid_path = default_vid_folder / Path(args.vid_name)
        elif args.vid_folder:
            vid_path = Path(args.vid_folder) / default_vid_name
        else:
            vid_path = default_vid_path
        self.vid_path = vid_path

        # Display
        self.display_bool = args.display

        # Console
        self.console = Console()

        # ---- Unpack other arguments ----
        self.show_graph = show_graph
        if draw_backdrop_plt_func:
            self.draw_backdrop_plt_func = draw_backdrop_plt_func
        if draw_graph_plt_func:
            self.draw_graph_plt_func = draw_graph_plt_func


    # =============================================================

    # ---- UTILITIES ---- 
    def iterate_all_alive_particles(self):
        """
        Generator to yield all alive particles.

        Yields:
            Particle: Different instances of child classes of Particle
        """
        for class_name, class_dict in self.state["Particle"].items():
            # TODO: Does list() change anything here?
            for particle in list(class_dict.values()):
                if particle.alive:
                    yield particle
    
    def iterate_all_particles(self):
        """
        Generator to yield all particles.

        Yields:
            Particle: Different instances of child classes of Particle
        """
        for class_name, class_dict in self.state["Particle"].items():
            for particle in list(class_dict.values()):
                yield particle
    
    def iterate_all_environments(self):
        """
        Generator to yield all environments.

        Yields:
            Environment: Different instances of child classes of Environment
        """
        for class_name, class_list in self.state["Environment"].items():
            for environment in list(class_list):
                yield environment

    def rich_progress(self, iterable, description: str = "Working", console=None):
        """
        Wraps an iterable with a Rich module progress bar that displays [iteration / total].

        Parameters:
            iterable (Iterable): The iterable object that we're looping over.
            description (str): Description label used by progress bar.

        Yields:
            Iterable elements plus progress bar.
        """
        # Get total length
        total = len(iterable) if hasattr(iterable, "__len__") else None
        
        # Get console singleton
        self.console = console or Console() #force_terminal=True

        # Make columns
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            TextColumn("[{task.completed}/{task.total}]"),  # Shows [n/total]
            BarColumn(),
            TaskProgressColumn(),  # Shows % complete
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(description, total=total)
            for item in iterable:
                yield item
                progress.update(task, advance=1)
    
    def print_settings_table(self):
        self.console = Console()
        long_line_divide = "[dim]" + "─" * 40 + "[/dim]"
        short_line_divide = "[dim]" + "─" * 20 + "[/dim]"
        table = Table(title="[bold]Selected Settings ⚙️[/bold]")
        table.add_column("Setting", justify="centre",style="bold magenta")
        table.add_column("Command Flag", justify="left",style="yellow")
        table.add_column("Value", justify="left", style="green")

        table.add_row("⚙️  Mode", "(argument 1)",f"{self.mode}")
        table.add_row("🦋 Simulation Type", "(argument 2)", f"{self.simulation_type}")
        table.add_row("🕹️  Interactive", "-i, --interactive", f"{self.interactive}")
        table.add_row(long_line_divide, short_line_divide, long_line_divide)

        table.add_row("💯 Total number of timesteps","-t, --steps", f"{self.num_steps}")
        table.add_row("⏳ Timestep duration (delta t)","-d, --deltat", f"{self.delta_t}")
        table.add_row(long_line_divide, short_line_divide, long_line_divide)

        table.add_row("🧮 Synchronous compute and render","-s, --sync", f"{self.sync_compute_and_rendering}")
        table.add_row("🧠 Cache history of timesteps","-c, --cache", f"{self.cache_history}")
        table.add_row("📝 Write to NDJSON log","-l, --log", f"{self.cache_history}")
        table.add_row("🎥 Save render to MP4 video","-v, --vid", f"{self.save_video}")
        table.add_row("🗄️  NDJSON log path", "--log_path\n OR --log_name\n OR --log_folder",f"{self.logger.log_path}")
        table.add_row("🎞️  MP4 video path", "--vid_path\n OR --vid_name\n OR --vid_folder", f"{self.vid_path}")

        print("")
        self.console.print(table)
        print("")

    # ---- STATE LOADING UTILITIES----
    def get_state_at_timestep(self, timestep: int):
        """
        Get a state dictionary from a given timestep.
        Behaviour depends on user supplied 'log' and 'cache' arguments:
        1. If available, loads from cached self.history list of state dicts.
        2. Else, loads from NDJSON file.

        Args:
            timestep (int): Timestep number

        Raises:
            Exception: If neither cache or log were written. (Shouldn't happen)

        Returns:
            dict: Nested state dictionary to use for setting Manager.state
        """
        if self.cache_history and not self.mode == "load":
            return self.history[timestep]
        elif self.write_log:
            return self.logger.read_log_at_timestep(timestep)
        else:
            raise Exception("Neither cache or log written but trying to read! in get_state_at_timestep!")

    def copy_state(self, new_state):
        """
        Recursively copies a new_state dictionary into Manager.state.
        Nested copying of each child instance in dict ensures continuity.

        Args:
            new_state (dict): Nested state dictionary
        """
        for key, val in new_state.items():
            if key == "Particle": # Uses dict
                for child_class_name, child_class_dict in val.items():
                    existing_child_class_dict = self.state[key][child_class_name]
                    for id, child in child_class_dict.items():
                        # Use copy_state to update if ID exists, otherwise deepcopy
                        if id in existing_child_class_dict.keys():
                            existing_child_class_dict[id].copy_state(child)
                        else:
                            self.state[key][child_class_name][id] = deepcopy(child)
                    # Remove remaining existing IDs if they dont appear
                    for id in existing_child_class_dict.keys():
                        if id not in child_class_dict.keys():
                            self.state[key][child_class_name].pop(id)

            elif key == "Environment": # Uses list
                for child_class_name, child_class_list in val.items():
                    # TODO: We assume number of Environment objects is static!
                    for index, child in enumerate(child_class_list):
                        existing_child_class_list = self.state[key][child_class_name]
                        existing_child_class_list[index].copy_state(child)
            else:
                # Assuming we can just copy
                self.state[key] = val

    # =============================================================

    # ---- MAIN PIPELINE OPERATIONS ----
    def start(self):
        """
        Called by simulation engine package's main entrypoint script.
        - Starts pipeline function for specified mode (ie run vs load).
        """
        if self.mode == 'run':
            self.run()
        elif self.mode == 'load':
            self.load()
    
    def run(self):
        """
        Main pipeline function for 'run' mode.

        0. Optionally records starting state
        1. If in asynchronous mode (default) computes all timesteps.
        2. Renders simulation timesteps with chosen rendering framework (eg matplotlib)
        """
        # 0. Record starting state
        if self.cache_history:
            self.history.append(deepcopy(self.state))
        if self.write_log:
            self.logger.append_current_state(self.state)

        # 1. Compute first if asynchronous
        if not self.sync_compute_and_rendering:
            for timestep in tqdm(range(self.num_steps), desc=f"Computation Progress", initial=1):
                self.update()
            print("")
            print("[cyan]Finished Computing![/cyan] 🥸")
            print("")
            # Reset to first step's state
            self.copy_state(self.get_state_at_timestep(0))
            self.current_time = 0
            self.current_step = 0
        
        # 2. Split by rendering framework
        if self.render_framework == "matplotlib":
            self.animate_plt()

    def load(self):
        """
        Main pipeline function for 'load' mode.

        1. Reads supplied log for simulation metadata used for running
        2. Renders simulation timesteps with chosen rendering framework (eg matplotlib)
        """
        # 1. Read log for metadata
        # Load the first state to get delta_t, env_x_lim, env_y_lim, track_com and torus
        self.state = self.get_state_at_timestep(0)
        # Get number of lines as num steps
        self.num_steps = len(self.logger.offset_indices) - 1

        # Settings
        self.sync_compute_and_rendering = False
        self.interactive = False

        # 2. Split by rendering framework
        if self.render_framework == "matplotlib":
            self.animate_plt()

    
    def update(self):
        """
        Main timestepping function for simulation engine.

        - Updates all particles in current Manager.state dictionary
        - Increments time and updates records
        - Optionally caches state dict by appending to self.history
        - Optionally writes state dict to NDJSON log
        """
        # Update all particles
        for particle in self.iterate_all_alive_particles():
            particle.update()

        # Increment time
        self.current_time += self.delta_t
        self.current_step += 1

        # Store state in history
        if self.cache_history:
            if self.sync_compute_and_rendering:
                '''
                Matplotlib Spines objects (axes) can't be deepcopied.
                If in sync mode, each particle/environment will have artists drawn, 
                and each artist references its parent axes. 
                So we reset plt.artists = None for each plottable object,
                and remove the artists from the existing axis.
                This is a regrettable work-around but avoids major architecture changes.
                '''
                for particle in self.iterate_all_particles():
                    particle.remove_from_plot_plt()
                for environment in self.iterate_all_environments():
                    environment.remove_from_plot_plt()

            self.history.append(deepcopy(self.state))

        # Write to file
        if self.write_log and not self.done_rendering:
            self.logger.append_current_state(self.state)

    # =============================================================
    
    # MATPLOTLIB PLOTTING

    # ---- MAIN ANIMATION PIPELINE ----
    def animate_plt(self):
        """
        Main animation pipeline function for rendering simulation via matplotlib.

        Does the following:
        1. Sets up figure and axes objects
        2. Creates a matplotlib FuncAnimation object to perform animation in window,
           which calls our draw_figure_plt function at each timestep
        3. Optionally saves video of the animation
        4. Renders video then plays on loop once cached.
        """
        # 1. Set up figure and axes
        fig, ax, ax2 = self.setup_figure_plt()

        # 2. Create a matplotlib FuncAnimation object which timesteps draw_figure_plt
        interval_between_frames = self.delta_t*1000 # milliseconds
        # Wrap the frames iterator in a custom rich progress bar
        frames_iterator = tqdm(range(self.num_steps),desc="Rendering Progress", initial=1)
        #frames_iterator = range(self.num_steps)
        # Tried blit=True with returned artists collected recursively by draw_figure_plt, couldn't get working.
        blit = False
        # Create a zero-argument init_func inside a scope that can see our desired arguments
        init_func_plt = self._make_init_func_plt(ax,ax2)
        # Compose object
        self.animation = FuncAnimation(fig=fig, 
                            func=self.draw_figure_plt,
                            init_func=init_func_plt,
                            frames=frames_iterator,
                            fargs=([ax],[ax2]), 
                            interval=interval_between_frames,
                            repeat=not self.save_video,
                            cache_frame_data=self.cache_history,
                            blit=blit)
        
        # 3. Split based on save video
        if self.save_video:
            print("Now rendering simulation frame by frame")
            # Make sure parent path exists
            self.vid_path.parent.mkdir(parents=True, exist_ok=True)
            fps = 1/self.delta_t # period -> frequency
            self.animation.save(self.vid_path.absolute().as_posix(), writer='ffmpeg', fps=fps)
            print("\n")
            print(f"Saved simulation as mp4 at {self.vid_path}.")
            # Display video
            if self.display_bool:
                self.display_rendered_video()
            else:
                plt.close(fig)
        else:
            print("Now rendering frames for each time step - [italic]not displaying real time[/italic]")
            print("")
            # 4. Play video on loop
            if self.display_bool:
                plt.show()
            else:
                plt.close(fig)        

    # ---- FIGURE SETUP ----
    def setup_figure_plt(self):
        """
        Set up matplotlib figure and axes objects

        Does the following:
        1. Optionally creates subplot figure with frame axes and graph axes
           Otherwise sets up single axes figure.
        2. Set tight layout with invisible ticks and axes
        3. Draws frame backdrop with user supplied (or default) function
        4. Adds title

        Returns:
            fig (plt.Figure): Main plot window figure 
            ax (plt.Axes): Main plot frame axes
            ax2 (plt.Axes): Optional graph axes
        """
        # 1. Set up figure and axes
        # Initialise as None
        fig, ax, ax2 = None, None, None

        if self.show_graph:
            # TODO: Should there be a setup graph function?
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
            max_time = int(self.num_steps)*self.delta_t
            ax2.set_xlim(0, max_time) 
        else:
            # Setup figure
            fig, ax = plt.subplots()#figsize=[20,15])
            height = 7 #in inches
            width = height * Particle.env_x_lim / Particle.env_y_lim
            fig.set_size_inches(width, height)

        # Initialise ax by plotting dimensions
        ax.set_xlim(xmin=0,xmax=Particle.env_x_lim)
        ax.set_ylim(ymin=0,ymax=Particle.env_x_lim)

        # 2. Set tight layout with invisible axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.set_aspect('equal', adjustable='box')
        fig.tight_layout()

        # 3. Draw backdrop with either default or user supplied backdrop
        self.draw_backdrop_plt_func(ax)

        # 4. Add title
        if self.done_rendering:
            window_title = f"Simulation Engine [{self.simulation_type}] | Step: {self.current_step}/{self.num_steps} | Time: {round(self.current_time,2)}s"
        else:
            window_title = f"RENDERING - Please do not quit! | Step: {self.current_step}/{self.num_steps}"
        fig.canvas.manager.set_window_title(window_title)

        return fig, ax, ax2
    
    def setup_frame_plt(self, ax:plt.Axes):
        """
        Setup main matplotlib frame axes object

        1. Optionally works out scene's centre of mass and scale
        2. Iterates through each child particle and environment instance and plots to ax

        Args:
            ax (plt.Axes): Main plot frame axes

        Returns:
            list: List of matplotlib 'artist' objects, mixed classes
        """
        # 1. Decide if tracking the COM in each frame
        com, scene_scale = None, None
        if Particle.track_com:
            com = Particle.centre_of_mass()
            scene_scale = Particle.scene_scale()

        # 2. Update particle and environment artists
        artists = []
        for environment_object_type, environment_objects_list in self.state["Environment"].items():
            for environment_object in environment_objects_list:
                artists += environment_object.draw_plt(ax, com,scene_scale)
        for particle in self.iterate_all_particles():
            artists += particle.draw_plt(ax, com,scene_scale)
        return artists
    
    def _make_init_func_plt(self, ax:list[plt.Axes], ax2:list[plt.Axes]=None):
        """
        Factory function to build init_func_plt, a zero-argument wrapper of setup_frame_plt.
        
        Matplotlib expects FuncAnimation's init_func to have zero arguments, 
        but we'd like to pass self.setup_frame_plt an 'ax' argument
        So we build a zero-argument wrapper function via a factory with ax in scope.

        Args:
            ax (list[plt.Axes]): Main simulation window's axis, inside list wrapper
            ax2 (list[plt.Axes], optional): Optional simulation window graph axis, inside list wrapper

        Returns:
            function: zero-argument wrapper function of setup_frame_plt
        """
        def init_func_plt():
            return self.setup_frame_plt(ax)
        return init_func_plt

    # ---- MAIN CALLABLE ----
    def terminate_on_error(method):
        """
        Decorator to stop Matplotlib FuncAnimation thread if error occurs.
        (Without this, an error would occur at each timestep and flood the terminal!)

        Args:
            animation (plt.animation.FuncAnimation): Main animation object used for simulation
        """
        def wrapper(self, *args, **kwargs):
            try:
                return method(self, *args, **kwargs)
            except Exception as e:
                print(f"Error in matplotlib animation update function: {e}")
                if hasattr(self, "animation") and self.animation:
                    self.animation.event_source.stop()
                plt.close("all")
                raise
        return wrapper
    
    @terminate_on_error
    def draw_figure_plt(self, timestep:int, ax:list[plt.Axes], ax2:list[plt.Axes]=None):
        """
        Main matplotlib timestep function called by a FuncAnimation object at each timestep.

        Does the following:
        1. Manages the loop metadata and tracks when we loop or finish rendering
        2. Loads the simulation state at the given timestep, (either through computation or loading)
        3. Calls the draw_frame_plt and draw_graph_plt functions, collecting matplotlib artists.

        Args:
            timestep (int): Timestep
            ax (list[plt.Axes]): Main simulation window's axis, inside list wrapper
            ax2 (list[plt.Axes], optional): Optional simulation window graph axis, inside list wrapper

        Returns:
            list: List of matplotlib 'artist' objects, mixed classes
        """
        # Unpack ax, ax2 from wrapper
        ax = ax[0]
        ax2 = ax2[0]

        # 1. Loop management
        self._loop_management_plt(timestep)

        # 2. Get the state
        self._load_state_at_timestep_plt(timestep)

        # Print current frame - this will usually be silenced by rich progress bar until rendering done
        if self.done_rendering:
            print(f"--> [italic]Displaying Frame[/italic][{self.current_step} / {self.num_steps}]" ,end="\r", flush=True)

        # 3. Draw frame and graph onto existing axes, return artists
        artists = []
        artists += self.draw_frame_plt(ax)
        if self.show_graph:
            artists += self.draw_graph_plt(ax2)
        
        return artists
    
    def _loop_management_plt(self, timestep):
        """
        Tracks when matplotlib FuncAnimation has finished rendering and loops,
        updating internal records.

        Args:
            timestep (int): Timestep
        """
        # Reset after loop
        if self.just_looped:
            if not self.done_rendering:
                self.done_rendering = True
                print("")
                print("[green]Finished Rendering![/green] 🐸")
                print("")
                print("Now displaying finished rendered frames in real time!")
                print("Press 'Q' in the plot window to exit.")
                print("")
            self.current_time = 0
            self.current_step = -1
            self.just_looped = False
        # Check for loop
        if timestep == self.num_steps-1:
            self.just_looped = True
    
    def _load_state_at_timestep_plt(self, timestep):
        """
        Loads the simulation state at the given timestep.
        Also manages 

        This will be computed if necessary,
        otherwise it will be loaded from the self.history cache, 
        or failing this the NDJSON log.

        Args:
            timestep (_type_): _description_
        """
        if self.sync_compute_and_rendering and \
            (not self.done_rendering or \
            (not self.cache_history and not self.write_log)):
            # Synchronous - we compute while rendering
            self.update()                
        else:
            # Asynchronous - we've already computed steps
            self.copy_state(self.get_state_at_timestep(timestep+1))
            # Update time
            self.current_step += 1
            self.current_time += self.delta_t

    # ---- DRAW FRAME ----
    def draw_frame_plt(self, ax:plt.Axes):
        """
        Draw all objects in the simulation's current state onto a supplied matplotlib ax object.

        1. Sets titles based on timestep
        2. Optionally works out scene's centre of mass and scale
        3. Iterates through each child particle and environment instance and plots to ax

        Args:
            ax (plt.Axes): Main plot frame axes

        Returns:
            list: List of matplotlib 'artist' objects, mixed classes
        """
        # 1. Add figure, ax titles
        normal_title = f"Simulation Engine [{self.simulation_type}] | Step: {self.current_step}/{self.num_steps} | Time: {round(self.current_time,2)}s"
        render_title = f"RENDERING - Please do not quit! | Step: {self.current_step}/{self.num_steps}"
        if self.done_rendering:
            window_title = normal_title
        else:
            window_title = render_title
        # Try setting figure title
        try:
            fig = ax.get_figure()
            fig.canvas.manager.set_window_title(window_title)
        except:
            pass
        ax.set_title(normal_title)

        # 2. Decide if tracking the COM in each frame
        com, scene_scale = None, None
        if Particle.track_com:
            com = Particle.centre_of_mass()
            scene_scale = Particle.scene_scale()

        # 3. Update particle and environment artists
        artists = []
        artists += self._draw_environment_objects_plt(ax, com, scene_scale)
        artists += self._draw_particle_objects_plt(ax, com, scene_scale)

        return artists

    @staticmethod
    def _default_draw_backdrop_plt(ax:plt.Axes):
        """
        Default function to draw a background onto a matplotlib axes.
        Ideally replaced by a particular simulation type's own function.

        Args:
            ax (plt.Axes): Main plot frame axes
        """
        # Default background for matplotlib frames if not specified by user
        ax.set_xlim(-1, Particle.env_x_lim+1)
        ax.set_ylim(-1, Particle.env_y_lim+1)

        # Black background
        ax.set_facecolor('k')
    
    draw_backdrop_plt_func = _default_draw_backdrop_plt

    def _draw_environment_objects_plt(self, ax:plt.Axes, com, scene_scale):
        """
        Draw all environment child objects onto a given matplotlib axes.

        Args:
            ax (plt.Axes): Main plot frame axes
            com (np.ndarray): 2D centre of mass of all Particle objects
            scene_scale (float): Size of range between furthest Particle objects

        Returns:
            list: List of matplotlib 'artist' objects, mixed classes
        """
        # Draw all environment objects from state dictionary onto supplied Matplotlib.pyplot ax object
        artists = []
        for environment_object_type, environment_objects_list in self.state["Environment"].items():
            for environment_object in environment_objects_list:
                artists += environment_object.draw_plt(ax, com,scene_scale)
        return artists
    
    def _draw_particle_objects_plt(self, ax:plt.Axes, com, scene_scale):
        """
        Draw all particle child objects onto a given matplotlib axes.

        Args:
            ax (plt.Axes): Main plot frame axes
            com (np.ndarray): 2D centre of mass of all Particle objects
            scene_scale (float): Size of range between furthest Particle objects

        Returns:
            list: List of matplotlib 'artist' objects, mixed classes
        """
        # Draw all particle objects from state dictionary onto supplied Matplotlib.pyplot ax object
        artists = []
        for particle in self.iterate_all_particles():
            artists += particle.draw_plt(ax, com,scene_scale)
        return []
    
    # ---- DRAW GRAPH ----
    def draw_graph_plt(self, ax2:plt.Axes):
        """
        Draw graph corresponding to current state onto given axes.

        Args:
            ax (plt.Axes): Main plot frame axes

        Returns:
            list: List of matplotlib 'artist' objects, mixed classes
        """
        # TODO: Add any shared functionality here
        # Currently just wrapping draw_graph_plt_func
        artists = []
        artists += self.draw_graph_plt_func(self, ax2)
        return artists

    @staticmethod
    def _default_draw_graph_plt(manager, ax2:plt.Axes):
        """
        Default function to draw graph corresponding to current state onto given axes.
        Ideally replaced by a particular simulation type's own function.

        Args:
            ax (plt.Axes): Main plot frame axes
        """
        artists = []
        max_time = int(manager.num_timesteps)*manager.delta_t
        ax2.set_xlim(0, max_time)  # Set x-axis limits
        return artists
    
    draw_graph_plt_func = _default_draw_graph_plt

    # ------------------------------------------------

    def display_rendered_video(self):
        # Set up a cv2 video capture instance from video path
        cap = cv2.VideoCapture(self.vid_path)
        # Check opened
        if not cap.isOpened():
            print(f"Error: Could not open rendered video at {self.vid_path}.")
            exit()
        # Create a named window (optional: make resizable)
        window_title = f"Simulation Engine [{self.simulation_type}] | {self.vid_path}"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

        # Main listening loop
        while True:
            # Read next frame
            ret, frame = cap.read()
            if not ret:
                # Restart video from the beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            # Show frame as image in window
            cv2.imshow(window_title, frame)

            # Press 'q' to quit early
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
        
        # Safely quit
        cap.release()
        cv2.destroyAllWindows()

# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------


class Logger:
    '''
    Logger utility class to handle writing/reading NDJSON logs
    Logs encode a Manager.state nested dictionary of Particle and Environment classes plus metadata

    '''
    # ---- INITIALISATION ---- 
    def __init__(self, manager, log_path:Path, chunk_size=100):
        """
        Initialise a Logger instance linked to a Manager instance, with log path and chunk size.

        Args:
            manager (Manager): The current manager controlling the workflow
            log_path (Path): Path to NDJSON log to read/write from
            chunk_size (int, optional): Number of lines to read at a time from NDJSON log. Defaults to 100.
        """
        self.log_path = log_path
        self.chunk_size = chunk_size
        self._offset_indices: list[int] = None
        self.manager: Manager = manager

        # Initialise log file
        if self.manager.write_log:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            # If in run mode, remove and start again
            if self.manager.mode == 'run':
                if self.log_path.exists():
                    os.remove(self.log_path)
                self.log_path.touch()
            # If in load mode, path must exist
            elif self.manager.mode == 'load' and not self.log_path.exists():
                raise FileExistsError(f"Log path {log_path} not found!")
    
    # =============================================================

    # ---- READ / WRITE STATE ----
    @staticmethod
    def write_state_to_dict(state):
        """
        Write a state dictionary to a compressed dictionary for writing to NDJSON.
        Iterates through all child Particle and Environment classes,
        and calls their to_dict() method.
        Note we may lose information depending on if the to_dict methods are lossy.

        Args:
            state (dict[dict]): Nested dictionary typically from Manager.state

        Returns:
            dict[dict]: Compressed nested dictionary to be appended to NDJSON log file
        """
        new_dict = {}
        for key, val in state.items():
            # Create nested dict for Particle and Environment using .to_dict() methods
            new_dict[key] = {}
            if key == "Particle":
                for child_class_name, child_class_dict in val.items():
                    new_dict[key][child_class_name] = {}
                    for id, child in child_class_dict.items():
                        new_dict[key][child_class_name][id] = child.to_dict()
            elif key == "Environment":
                for child_class_name, child_class_list in val.items():
                    new_dict[key][child_class_name] = []
                    for child in child_class_list:
                        new_dict[key][child_class_name].append(child.to_dict())
            # Assuming all other keys are simple
            else:
                new_dict[key]=val
        return new_dict
    
    def load_state_from_dict(self, json_dict):
        """
        Load a state dictionary from a compressed dictionary read from an NDJSON log file.
        Iterates through all child Particle and Environment classes,
        and calls their from_dict() method.

        Args:
            dict (dict[dict]): Compressed nested dictionary read from NDJSON log file

        Returns:
            dict: Nested state dictionary to be used to create a Manager.state dict
        """
        new_state = {}
        for key, val in json_dict.items():
            # Create nested dict for Particle and Environment using .to_dict() methods
            new_state[key] = {}
            if key == "Particle":
                for child_class_name, child_class_dict in val.items():
                    child_class = self.manager.class_objects_registry[child_class_name]
                    new_state[key][child_class_name] = {}
                    for id, child_dict in child_class_dict.items():
                        new_state[key][child_class_name][int(id)] = child_class.from_dict(child_dict)
            elif key == "Environment":
                for child_class_name, child_class_list in val.items():
                    child_class = self.manager.class_objects_registry[child_class_name]
                    new_state[key][child_class_name] = []
                    for child_dict in child_class_list:
                        new_state[key][child_class_name].append(child_class.from_dict(child_dict))
            # Assuming all other keys are simple
            else:
                new_state[key]=val
        # Unpack metadata to manager
        self.simulation_type = json_dict["type"]
        self.delta_t = json_dict["delta_t"] 
        self.current_step = json_dict["step"]
        self.current_time = json_dict["time"] 
        Particle.env_x_lim = json_dict["env_x_lim"]
        Particle.env_y_lim = json_dict["env_y_lim"]
        Particle.track_com = json_dict["track_com"]
        Particle.torus = json_dict["torus"]
        
        return new_state
    
    def append_current_state(self, state):
        """
        Append a state dictionary to the end of the current NDJSON log file.
        Calls on Logger.write_state_to_dict(state)

        Args:
            state (dict): Nested state dictionary from Manager.state
        """
        # Write main dictionary
        new_dict = self.write_state_to_dict(state)

        # Add metadata
        new_dict["type"] = self.manager.simulation_type
        new_dict["delta_t"] = self.manager.delta_t
        new_dict["step"] = self.manager.current_step
        new_dict["time"] = self.manager.current_time
        new_dict["env_x_lim"] = Particle.env_x_lim
        new_dict["env_y_lim"] = Particle.env_y_lim
        new_dict["track_com"] = Particle.track_com
        new_dict["torus"] = Particle.torus

        # Append to NDJSON
        with open(self.log_path, "a") as f:
            f.write(json.dumps(new_dict)+"\n")

    # =============================================================

    # ---- ACCESS PARTICULAR TIMESTEP ----
    @property
    def offset_indices(self):
        """
        Getter to generate a list of byte offsets for NDJSON file:
        - Reads self.log_path and records each offset when a new line (ie new state dict) is reached.
        - Stores list of these byte offset indices to self.offset_indices
        """
        if self._offset_indices is None:
            offset_indices = []
            with open(self.log_path, "r") as f:
                # While not done keep writing offsets
                while True:
                    # Get cursor position with tell
                    offset = f.tell()
                    # Read a line, break if nothing there
                    line = f.readline()
                    if not line:
                        break
                    # Add offset to list
                    offset_indices.append(offset)
            self._offset_indices = offset_indices
        return self._offset_indices

    def read_log_at_timestep(self, timestep:int):
        """
        Read the NDJSON log for the state at given timestep,
        return nested state dictionary for setting Manager.state

        Args:
            timestep (int): Timestep of the simulation to load

        Returns:
            dict[dict]: Nested state dictionary from Manager.state
        """
        with open(self.log_path, "r") as f:
            # Go to cursor offset of desired line
            f.seek(self.offset_indices[timestep])
            # Read the line into a JSON string
            line_str = f.readline()
        # Read the JSON string into a compressed state dict
        line_dict = json.loads(line_str)
        # Load a state dictionary from this compressed state dict
        state = self.load_state_from_dict(line_dict)
        return state
    
    # =============================================================

    # ---- YIELD BY CHUNKING ----
    def iter_all_states(self):
        """
        Generator to read all states from an NDJSON in chunks.
        Loops through chunks of size self.chunk_size,
        reads each chunk's JSON lines and yields states

        Yields:
            state: Nested state dictionary to be used to set Manager.state
        """
        for chunk in self._read_by_chunk():
            for line in chunk:
                yield self.load_state_from_dict(line)

    def _read_by_chunk(self):
        """
        Generator to loop through chunks of lines in an NDJSON file.
        Yields up to self.chunk_size many JSON lines from the file.

        Yields:
            list[dict]: Chunk list of nested state dictionaries
        """
        with open(self.log_path, "r") as f:
            # Start with empty chunk list
            chunk = []
            # Iterate over lines
            for line in f:
                # Read each line as JSON string, store in chunk
                chunk.append(json.loads(line))
                # Stop at chunk size and yield, clear chunk
                if len(chunk) == self.chunk_size:
                    yield chunk
                    chunk = []
            # Yield final partial chunk
            if chunk:
                yield chunk