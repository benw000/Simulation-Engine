import numpy as np

from simulation_engine.utils.errors import SimulationEngineInputError
from simulation_engine.classes.parents import Particle, Wall, Target
from simulation_engine.utils.manager import Manager

def setup(args):
    """
    Called by main entrypoint script as entry into this simulation 'type' module.
    Divides between different setup functions for each different 'mode'

    Args:
        args (argparse.Namespace): argparse namespace of user supplied arguments
    """
    # Create manager instance
    manager = Manager(args = args, 
                      show_graph = False,
                      draw_backdrop_plt_func = draw_backdrop_plt)
    
    # Add Star child class to registry
    manager.class_objects_registry["Pinball"] = Pinball
    manager.class_objects_registry["Wall"] = Wall 
    manager.class_objects_registry["Target"] = Target

    # Split by mode
    if args.mode == 'run':
        return setup_run(args, manager)
    elif args.mode == 'load':
        return manager

def setup_run(args, manager):
    # Validate args
    if not len(args.nums) == 1:
        raise SimulationEngineInputError("(-n, --nums) Please only supply 1 argument for population when using nbody simulation type")
    
    # Set Particle geometry attributes
    x, y = 10, 10
    Particle.env_x_lim = x
    Particle.env_y_lim = y

    # Initialise particles - could hide this in Particle but nice to be explicit
    num_stars = args.nums[0]
    # Generate a random set of pinballs
    for i in range(num_stars):
        Pinball()
    
    # Border walls
    Wall(np.array([0,0]),np.array([x,0]))
    Wall(np.array([0,0]),np.array([0,y]))
    Wall(np.array([x,0]),np.array([x,y]))
    Wall(np.array([0,y]),np.array([x,y]))
    

    return manager

def draw_backdrop_plt(ax):
    """
    Get an ax from manager, and plot things on it related to this mode
    Overrides Manager.default_draw_backdrop_plt

    Args:
        ax (plt.Axes): Main matplotlib frame
    """
    # Black background
    ax.set_facecolor('white')

class Pinball(Particle):
    # Force constants
    ball_repulsion = 2
    radius = 1

    wall_const_A = 200
    wall_const_B = 2

    gravity = np.pi**2
    
    # Initialisation
    def __init__(self, position: np.ndarray = None, velocity: np.ndarray = None, unlinked: bool=False) -> None:
        '''
        Initialises a Pinball object, inheriting from the Particle class.
        '''
        super().__init__(position, velocity, unlinked)
        # Put extra code here

    # -------------------------------------------------------------------------
    # Main force model 

    def update_acceleration(self):
        '''
        Calculates main acceleration term from force-based model of environment.
        '''
        # Instantiate force term
        force_term = np.zeros(2)

        # Ball-ball repulsion force
        for ball in self.__class__.iterate_class_instances():
            tol = 1e-3
            if ball == self:
                continue
            elif self.dist(ball) < self.radius + tol:
                force_term += - self.unit_dirn(ball)*(self.ball_repulsion/(np.sqrt(self.dist(ball))))

        # Repulsion from walls - exponential scaling 
        # (Helbing and Molnar 1995)
        for wall in self.manager.state["Environment"]["Wall"]:
            dist, dirn = wall.dist_to_wall(self.position)
            if dist < self.wall_const_B:
                force_term += dirn * self.wall_const_A * np.exp(-(np.sqrt(dist)/self.wall_const_B))

        # Gravity
        force_term += np.array([0,-self.gravity])

        # Update acceleration = Force / mass
        self.acceleration = force_term / self.mass

        return 0
    
    # -------------------------------------------------------------------------
    # Logging

    def to_dict(self):
        new_dict = super().to_dict()
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

    def draw_plt(self, ax, com=None, scale=None):
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




