import numpy as np
import matplotlib.pyplot as plt

from simulation_engine.classes import Particle, Wall, Target
from simulation_engine.utils.manager import Manager

# -------------------------------------------------------------------------
# Setup

def setup(args):
    """
    Called by main entrypoint script as entry into this simulation 'type' module.
    Divides between different setup functions for each different 'mode'

    Args:
        args (argparse.Namespace): argparse namespace of user supplied arguments
    """
    # Default timestep
    if args.deltat is None:
        args.deltat=Human.DEFAULT_TIMESTEP

    # Create manager instance
    manager = Manager(args = args, 
                      show_graph = True,
                      draw_backdrop_plt_func = draw_backdrop_plt,
                      draw_graph_plt_func = draw_graph_plt)
    
    # Add Prey, Predator child classes to registry
    manager.class_objects_registry["Human"] = Human
    manager.class_objects_registry["Wall"] = Wall 
    manager.class_objects_registry["Target"] = Target

    # Split by mode
    if args.mode == 'run':
        return setup_run(args, manager)
    elif args.mode == 'load':
        return manager
    
def setup_run(args, manager):
    # Set Particle geometry attributes
    Particle.env_x_lim = 12
    Particle.env_y_lim = 10
    Particle.track_com = False
    Particle.torus = False
    
    # Initialise walls
    x = Particle.env_x_lim
    y = Particle.env_y_lim
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
    Target(np.array([x+1,2.5]))
    Target(np.array([x+1,7.5]))

    # Spawn in evacuees
    for i in range(args.nums[0]):
        Human()

    # Initialise statistics
    manager.state["num_evacuees"] = args.nums[0]
    manager.state["num_escaped"] = 0

    return manager

def draw_backdrop_plt(ax):
    """
    Get an ax from manager, and plot things on it related to this mode
    Overrides Manager.default_draw_backdrop_plt

    Args:
        ax (plt.Axes): Main matplotlib frame
    """
    # White background
    ax.set_facecolor('w')
    return ax

def draw_graph_plt(manager, ax2):
    """
    Function to update the graph on ax2 with latest information.

    - Stores the graph plt artists in Human.plt_graph_artists list attribute
    - These artists contain graph history, which is appended to,
      allowing us to draw a dynamic graph with a history trail.
    - Relies on plt artist methods get_offsets() to access stored history

    Args:
        manager (Manager): The current manager instance
        ax2 (plt.Axes): The matplotlib axes which the graph is plotted on

    Returns:
        list: List of plt artists (used for blitting)
    """
    # Reset
    if manager.current_step == 0:
        Human.plt_graph_artists = None

    if Human.plt_graph_artists is None:
        # Set up the graph at the start
        ax2.clear()
        max_time = int(manager.num_steps)*manager.delta_t
        ax2.set_xlim(0, max_time)  # Set x-axis limits
        ax2.set_ylim(0, manager.state["num_evacuees"]) 
        xticks = [i for i in range(int(max_time)) if i % 5 == 0] + [max_time]  # Positions where you want the labels
        ax2.set_xticks(xticks)  # Set ticks at every value in the range
        ax2.set_xlabel("Time (s)")
        ax2.set_title(f"Number evacuated over time")
        ax2.set_aspect(aspect=Particle.env_y_lim/Particle.env_x_lim)

        # Get starting data
        y = manager.state["num_escaped"]
        x = manager.current_time
        # Initialise artists with scatter and plot
        Human.plt_graph_artists = [
            plt.plot([x],[y], c='b')[0], # Plot cumulative history
            plt.scatter([x],[y], marker='x', c='r') # Plot current point
        ]
    else:
        # Get current data from plt.plot
        x_data, y_data = Human.plt_graph_artists[0].get_data()
        # Get latest data
        y = manager.state["num_escaped"]
        x = manager.current_time
        # Append
        x_data = np.hstack((x_data, x))
        y_data = np.hstack((y_data, y))
        # Replot
        Human.plt_graph_artists[0].set_data(x_data, y_data)
        Human.plt_graph_artists[1].set_offsets([x,y])

    return Human.plt_graph_artists
        

class Human(Particle):
    '''
    Human particle for crowd simulation.
    '''
    # -------------------------------------------------------------------------
    # Force model constants

    # Default timestep
    DEFAULT_TIMESTEP = 0.05

    # Personal space Human-Human repulsion
    personal_space = 0.5 # metres - 2 rulers between centres
    personal_space_repulsion = 300 # Newtons
    # Wall repulsion/deflection
    wall_dist_thresh = 0.5
    wall_repulsion = 2000
    wall_deflection = 3000
    # Constant force from each human to their target
    target_attraction = 200
    random_force = 100

    # Tracking statistics
    num_evacuees = None
    num_escaped = 0
    plt_graph_artists = None
    
    # Initialisation
    def __init__(self, position: np.ndarray = None, velocity: np.ndarray = None, unlinked=None) -> None:
        '''
        Initialises a Human, inheriting from the Particle class.
        '''
        super().__init__(position, velocity, unlinked)

        # Human specific attributes
        self.mass = 50
        self.max_speed = 1.5

        # Imprint on nearest target
        self.my_target = Target.find_closest_target(self)

    # -------------------------------------------------------------------------
    # Distance utilities

    def wall_deflection_dirn(self, wall):
        """
        Create an arbitrary force term to repel Human away from walls, in direction of target.

        Args:
            wall (Wall): Wall that the Human is within distance threshold of

        Returns:
            np.ndarray (2,): Force term
        """
        '''
        When the wall isnt normal to the line from Human to Target, this is easy:
            - We get the angle between this normal and the Human's desired line
            - Depending on positive or negative angle, we push the Human +-the wall vector direction along the wall
            - This makes them shuffle/deflect along the wall while still being repelled by it as well.
        When the wall is normal:
            - We can have the Human headbutting the wall, not knowing which way to go.
            - Ideally a proper evacuation sim would incorporate elements of pathfinding.
        Deflection against the wall is one of the reasons why a purely force-based model fails,
        since we need to incorporate increasingly complicated heuristic forces to replicate intelligent behaviour.
        '''
        # Get distance and direction to current target
        wall_dist, wall_dirn = wall.dist_to_wall(self.position)
        wall_normal = wall.perp_vec
        target_dist, target_dirn = self.dist(self.my_target, return_both=True)
        # Get the angle between the particle<->target line and the normal to the wall
        angle = np.arccos(np.dot(wall_normal,target_dirn)/(np.linalg.norm(wall_normal)*np.linalg.norm(target_dirn)))
        # Tolerance to not apply force when in line with wall
        tolerance = 1e-6
        # Fudge factor to make Human decisive - no bouncing around equillibrium of zero
        fudge = 0.2
        if angle > ((-np.pi / 2) + tolerance) and angle < fudge:
            force_dirn = np.matmul(np.array([[0,-1],[1,0]]),wall_dirn)/wall_dist
            return force_dirn * self.wall_deflection # * np.cos(0.25*angle)
        elif angle >= fudge and angle<np.pi/2 - tolerance:
            force_dirn = np.matmul(np.array([[0,1],[-1,0]]),wall_dirn)/wall_dist
            return force_dirn * self.wall_deflection # * np.cos(0.25*angle)
        else:
            return np.zeros(2)

    # -------------------------------------------------------------------------
    # Main force model 

    def update_acceleration(self):
        '''
        Calculates main acceleration term from force-based model of environment.
        '''
        # Reconsider target every 20 timesteps
        if self.manager.current_step % 10 == 0:
            self.my_target = Target.find_closest_target(self)

        # Instantiate force term
        force_term = np.zeros(2)

        # Go through targets and check distance to escape threshold
        # If escape possible, unalive self. Otherwise sum my_target's force contribution
        for target in self.manager.state["Environment"]["Target"]:
            dist, dirn = self.dist(target, return_both=True)
            if dist < target.capture_thresh:
                self.manager.state["num_escaped"] += 1
                self.unalive()
                return 1
            elif target is self.my_target:
                force_term += self.target_attraction * dirn #/dist

        # Human repulsion force - currently scales with 1/d
        for human in Human.iterate_class_instances():
            if human == self:
                continue
            elif self.dist(human) < self.personal_space:
                force_term += - self.unit_dirn(human)*(self.personal_space_repulsion/(np.sqrt(self.dist(human))))
                pass

        # Repulsion from walls - scales with 1/d^2
        for wall in self.manager.state["Environment"]["Wall"]:
            dist, dirn = wall.dist_to_wall(self.position)
            if dist < self.wall_dist_thresh:
                force_term += dirn * (self.wall_repulsion/(dist))
                # Make Humans smart - repel sideways if vector to target is directly blocked by wall
                if dist < self.wall_dist_thresh: #*0.5:
                    force_term += self.wall_deflection_dirn(wall)

        # Random force - stochastic noise
        # Generate between [0,1], map to [0,2] then shift to [-1,1]
        force_term += ((np.random.rand(2)*2)-1)*self.random_force

        # Update acceleration = Force / mass
        self.acceleration = force_term / self.mass

        return 0
    
    # -------------------------------------------------------------------------
    # Logging

    def to_dict(self):
        new_dict = super().to_dict()
        new_dict["max_speed"] = self.max_speed
        return new_dict
    
    @classmethod
    def from_dict(cls, dict):
        instance = cls(position=np.array(dict["position"]),
                   velocity=np.array(dict["velocity"]),
                   unlinked=True)
        instance.acceleration = np.array(dict["acceleration"])
        instance.last_position = np.array(dict["last_position"])
        instance.mass = dict["mass"]
        instance.alive = dict["alive"]

        return instance

    # -------------------------------------------------------------------------
    # Matplotlib

    def draw_plt(self, ax:plt.Axes, com=None, scale=None):
        '''
        Updates the stored self.plt_artist PathCollection with new position
        '''
        # Remove artist from plot if dead
        if not self.alive:
            return self.remove_from_plot_plt()
        
        # Get plot position in frame
        plot_position = self.orient_to_com(com, scale)

        # Update artist with PathCollection.set_offsets setter method
        if self.plt_artists is None:
            # Initialise PathCollection artist as scatter plot point
            self.plt_artists = []
            self.plt_artists.append(ax.scatter(plot_position[0],plot_position[1], s=12**2,c='b'))
        else:
            # Update with offset
            self.plt_artists[0].set_offsets(plot_position)
        
        return self.plt_artists