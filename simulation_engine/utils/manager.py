import json
from rich import print
from rich.table import Table
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TaskProgressColumn
from copy import deepcopy
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

from simulation_engine.classes.parents import Particle, Environment, Wall, Target
from simulation_engine.utils.errors import SimulationEngineInputError


class Manager:
    '''
    Manages the collection of particle and environment entities
    '''
    # ---- INITIALISATION ---- 
    def __init__(self,
                 args:argparse.Namespace,
                 show_graph:bool = False,
                 draw_backdrop_plt_func = None,
                 ):
        """
        Initialise a Manager object to oversee the timestepping and state storing
        This will be called by a simulation type's setup() function,
        but is only completed after initialisation at the end of the setup() function.
        (After this, the main entrypoint script will call the Manager.run() method.)
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

        self.done_computing = False
        self.just_looped = False

        # ---- Unpack entrypoint args ----
        # Mode arguments
        self.mode = args.mode
        self.simulation_type = args.type
        self.interactive = args.interactive
        # TODO: Decide on render framework to use from these
        self.render_framework = "matplotlib"

        # Time
        self.num_steps: int = args.steps
        self.delta_t: float = args.deltat
        self.current_time = 0
        self.current_step = 0

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
        # Such nice logic 🥹
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

        # ---- Unpack other arguments ----
        self.show_graph = show_graph
        if draw_backdrop_plt_func:
            self.draw_backdrop_plt_func = draw_backdrop_plt_func


    # =============================================================

    # ---- UTILITIES ---- 
    def iterate_all_particles(self):
        for class_name, class_dict in self.state["Particle"].items():
            for id, particle in class_dict.items():
                if particle.alive:
                    yield particle

    def rich_progress(self, iterable, description: str = "Working"):
        """
        Wraps an iterable with a Rich progress bar that displays [iteration / total].

        Parameters:
            iterable (Iterable): The iterable object that we're looping over.
            description (str): Description label used by progress bar.

        Yields:
            Iterable elements plus progress bar.
        """
        # Get total length
        total = len(iterable) if hasattr(iterable, "__len__") else None
        
        # Make columns
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            TextColumn("[{task.completed}/{task.total}]"),  # Shows [n/total]
            BarColumn(),
            TaskProgressColumn(),  # Shows % complete
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(description, total=total)
            for item in iterable:
                yield item
                progress.update(task, advance=1)
        
        
    # =============================================================

    # ---- MAIN OPERATIONS ----
    def start(self):
        """
        Called by main entrypoint script, divides up between modes.
        Each mode has its own pipeline function.
        """
        # Display settings
        console = Console()
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

        table.add_row("💯 Total number of timesteps","-t, --steps" f"{self.num_steps}")
        table.add_row("⏳ Timestep duration (delta t)","-d, --deltat", f"{self.delta_t}")
        table.add_row(long_line_divide, short_line_divide, long_line_divide)

        table.add_row("🧮 Synchronous compute and render","-s, --sync", f"{self.sync_compute_and_rendering}")
        table.add_row("🧠 Cache history of timesteps","-c, --cache", f"{self.cache_history}")
        table.add_row("📝 Write to NDJSON log","-l, --log", f"{self.cache_history}")
        table.add_row("🎥 Save render to MP4 video","-v, --vid", f"{self.save_video}")
        table.add_row("🗄️  NDJSON log path", "--log_path\n OR --log_name\n OR --log_folder",f"{self.logger.log_path}")
        table.add_row("🎞️  MP4 video path", "--vid_path\n OR --vid_name\n OR --vid_folder", f"{self.vid_path}")

        print("")
        console.print(table)
        del console

        if self.mode == 'run':
            self.run()
        elif self.mode == 'load':
            self.load()
    
    def run(self):
        # Compute first if asynchronous
        if not self.sync_compute_and_rendering:
            for timestep in self.rich_progress(range(self.num_steps), description=f"[bold]Computation Progress[/bold]"):
                # print(f"----- Computation Progress: {self.current_step} / {self.num_steps} -----" ,end="\r", flush=True)
                self.update()
            print("")
            print("[cyan]Finished Computing![/cyan] 🥸")
            print("")
            # Reset to first step's state
            # self.state = self._read_state_at_timestep(0)
            self.current_time = 0
            self.current_step = 0 # -1
        


        # Split by rendering framework
        if self.render_framework == "matplotlib":
            self.animate_plt()

    def load(self):
        raise NotImplementedError
    
    def update(self):
        """
        To be called at each simulation timestep
        """
        # Update all particles
        for particle in self.iterate_all_particles():
            particle.update()

        # Increment time
        self.current_time += self.delta_t
        self.current_step += 1

        # TODO: Update kill records?

        # Store state in history
        if self.cache_history:
            self.history.append(deepcopy(self.state))

        # Write to file
        if self.write_log and not self.done_computing:
            self.logger.append_current_state(self.state)

    def load_state_at_timestep(self, timestep):
        # Reset after loop
        if self.just_looped:
            if not self.done_computing:
                self.done_computing = True
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

        # Either compute or read state
        if self.sync_compute_and_rendering and \
            (not self.done_computing or \
            (not self.cache_history and not self.write_log)):
            # Synchronous - we compute while rendering
            self.update()                
        else:
            # Asynchronous - we've already computed steps
            self.update_state(self._read_state_at_timestep(timestep))
            self.current_step += 1
            self.current_time += self.delta_t
    
    def _read_state_at_timestep(self, timestep):
        # Try cached history, then from log file
        if self.cache_history:
            return self.history[timestep]
        elif self.write_log:
            return self.logger.get_state_at_timestep(timestep)
        else:
            raise Exception("Neither cache or log written but trying to read! in _read_state_at_timestep!")

    def update_state(self, new_state):
        # TODO: write comment about why this is needed, deepcopy issue
        for key, val in self.state.items():
            if key in ["Particle", "Environment"]:
                if key == "Particle":
                    for child_class_name, child_class_dict in val.items():
                        for id, child in child_class_dict.items():
                            new_object = new_state[key][child_class_name][id]
                            child.update_state(new_object)
                elif key == "Environment":
                    for child_class_name, child_class_list in val.items():
                        for idx, child in enumerate(child_class_list):
                            new_object = new_state[key][child_class_name][idx]
                            child.update_state(new_object)
            # Assuming all other keys are simple, not objects
            else:
                self.state[key] = new_state[key]


    # =============================================================
    
    # ---- MATPLOTLIB PLOTTING ----

    # ---- ANIMATION ----
    def animate_plt(self):
        # Set up figure
        fig, ax, ax2 = self.setup_figure_plt()

        print("Now rendering frames for each time step - [italic]not displaying real time[/italic]")
        print("")

        # Setup a matplotlib FuncAnimation of draw_figure_plt
        # (draw_figure_plt handles updating/loading state)
        interval_between_frames = self.delta_t*1000 # milliseconds
        frames_iterator = self.rich_progress(range(self.num_steps),description="[bold]Rendering Progress[/bold]")
        animation = FuncAnimation(fig=fig, 
                            func=self.draw_figure_plt,
                            frames=frames_iterator,
                            fargs=([ax],[ax2]), 
                            interval=interval_between_frames,
                            repeat=True,
                            cache_frame_data=self.cache_history)
        
        # Optionally save video
        if self.save_video:
            # Make sure parent path exists
            self.vid_path.parent.mkdir(parents=True, exist_ok=True)
            fps = 1/self.delta_t # period -> frequency
            animation.save(self.vid_path.absolute().as_posix(), writer='ffmpeg', fps=fps)
            print("\n")
            print(f"Saved simulation as mp4 at {self.vid_path}.")
        
        # Play video on loop
        plt.show()

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
            height = 7 #in inches
            width = height * Particle.env_x_lim / Particle.env_y_lim
            fig.set_size_inches(width, height)

        # Initialise ax by plotting dimensions
        ax.set(xlim=[0,Particle.env_x_lim], ylim=[0,Particle.env_x_lim])
        
        # Add title
        if self.done_computing:
            window_title = f"Simulation Engine [{self.simulation_type}] | Step: {self.current_step}/{self.num_steps} | Time: {round(self.current_time,2)}s"
        else:
            window_title = f"RENDERING - Please do not quit! | Step: {self.current_step}/{self.num_steps}"
        fig.canvas.manager.set_window_title(window_title)

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
        # Get the right state
        self.load_state_at_timestep(timestep)

        # Print current frame - this will usually be silenced by rich progress bar until rendering done
        print(f"--> [italic]Displaying Frame[/italic][{self.current_step} / {self.num_steps}]" ,end="\r", flush=True)

        # Unpack ax, ax2 from wrapper
        ax = ax[0]
        ax2 = ax2[0]

        # Draw frame and graph onto existing axes
        self.draw_frame_plt(ax)
        if self.show_graph:
            self.draw_graph_plt(ax2)

    # ---- DRAW FRAME ----
    def draw_frame_plt(self, ax):
        '''
        Draw all objects in the simulation's current state onto a supplied matplotlib ax object.
        '''
        
        # Add title
        if self.done_computing:
            window_title = f"Simulation Engine [{self.simulation_type}] | Step: {self.current_step}/{self.num_steps} | Time: {round(self.current_time,2)}s"
        else:
            window_title = f"RENDERING - Please do not quit! | Step: {self.current_step}/{self.num_steps}"
        fig = ax.get_figure()
        fig.canvas.manager.set_window_title(window_title)

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


# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------


class Logger:
    '''
    TODO: Docstring here
    Also remove all the csv
    Split up these functions into readable structure
    '''
    # =============================================================

    # ---- INITIALISATION ---- 
    def __init__(self, manager, log_path:Path, chunk_size=100):
        self.log_path = log_path
        self.chunk_size = chunk_size
        self.offset_indices: list[int] = []
        self.manager = manager

        # Initialise log file
        if self.manager.write_log:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.touch()
    
    # =============================================================

    # ---- READ / WRITE STATE ----
    @staticmethod
    def write_dict_from_state(state):
        # Read state nested dict and create JSON format state
        # This should iterate through everything and call its return dictionary functions
        new_dict = {}
        for key, val in state.items():
            # Create nested dict for Particle and Environment using .to_dict() methods
            if key in ["Particle", "Environment"]:
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
    
    def load_state_from_dict(self, dict):
        new_state = {}
        for key, val in dict.items():
            # Create nested dict for Particle and Environment using .to_dict() methods
            if key in ["Particle", "Environment"]:
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
        return new_state
    
    # =============================================================

    # ---- APPEND ----
    def append_current_state(self, state):
        new_dict = self.write_dict_from_state(state)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(new_dict)+"\n")

    # =============================================================

    # ---- YIELD BY CHUNKING ----
    def iter_all_states(self):
        # Loops through chunks, reads through lines and yields states
        for chunk in self._read_by_chunk():
            for line in chunk:
                yield self.load_state_from_dict(line)

    def _read_by_chunk(self):
        # Reads through final and yields chunks
        # Read chunk_size many lines if possible from file
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

    # =============================================================

    # ---- ACCESS PARTICULAR TIMESTEP ----
    def _build_offset_indices(self):
        # Read the log path and go through, note down byte offsets
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
        self.offset_indices = offset_indices

    def get_state_at_timestep(self, timestep:int):
        # Get offset indices
        if self.offset_indices == []:
            self._build_offset_indices()
        # Read file at specific location
        with open(self.log_path, "r") as f:
            # Go to cursor offset of desired line
            f.seek(self.offset_indices[timestep])
            # Read the line into state
            line_str = f.readline()
        return self.load_state_from_dict(json.loads(line_str))
    

'''
TODO:
Subtle issue
When we append to history, we append self.state
But then we get a history full of the same exact self.state object, and each
part of the self.history list updates with each timestep

Instead we want to use deepcopy(self.state).
But then the artists inside each have not been initialised
So they must be initialised - this is solved
But then because each history state is a different object, each artist is different
So we track more and more artists
When really we want the same artists to be throughout

So we need a self.state update function!
This takes an object from history and makes the relevant changes
    


'''




'''
TODO: 
Look through git changes
It doesnt work, and it used to. Get this back.

Then fix the whole of the FuncAnimation and iter stuff.
Want a window to pop up, want the video to work, getting silly

Need it to stop calculating at a certain point, and just behave like normal.
So frustrating

'''