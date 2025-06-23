import numpy as np
import matplotlib.pyplot as plt

from simulation_engine.utils.errors import SimulationEngineInputError
from simulation_engine.core.particle import Particle
from simulation_engine.core.manager import Manager

# -------------------------------------------------------------------------
# Setup

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
    manager.class_objects_registry["Star"] = Star

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
    Particle.env_x_lim = 1000
    Particle.env_y_lim = 1000
    Particle.track_com = True
    Particle.torus = False

    # Initialise particles - could hide this in Particle but nice to be explicit
    num_stars = args.nums[0]
    random = True
    if random:
        # Generate a random set of stars
        for i in range(num_stars):
            Star()
    else:
        # Hardcode a specific set of stars - 3 body problem
        Star(np.array([300,300]),np.array([200,100]))
        Star(np.array([600,500]),np.array([-100,300]))
        Star(np.array([400,700]),np.array([0,-100]))

    return manager
    
def draw_backdrop_plt(ax):
    """
    Get an ax from manager, and plot things on it related to this mode
    Overrides Manager.default_draw_backdrop_plt

    Args:
        ax (plt.Axes): Main matplotlib frame
    """
    # Black background
    ax.set_facecolor('k')


class Star(Particle):
    '''
    Star particle for N-body simulation
    '''
    
    G = 5000
    min_mass = 10 
    max_mass = 10**3

    random_force = 0
    
    # Initialisation
    def __init__(self, position: np.ndarray = None, velocity: np.ndarray = None, unlinked: bool=False) -> None:
        '''
        Initialises a star object, inheriting from the Particle class.
        '''
        super().__init__(position, velocity, unlinked)
        
        # Get mass from distribution
        self.mass = self.get_log_uniform_mass()
        # Get size from exponential function of mass
        self.size = 5 + 10 * (np.power(2,np.log10(self.mass))-1)
        # Get velocity from 1/mass * 10 * random direction
        self.velocity = 10*np.array([np.random.rand(1)[0]*2 - 1,np.random.rand(1)[0]*2 - 1])

        # Random gray colour for plotting between 0.5 and 1
        self.colour = np.random.rand()/2 + 0.5

    # -------------------------------------------------------------------------
    # Utils

    def get_log_uniform_mass(self):
        '''
        Get a mass sampled from a log uniform distribution between
        Star.min_mass and Star.max_mass
        '''
        rng = np.random.default_rng()
        return np.exp(rng.uniform(np.log(self.__class__.min_mass), np.log(self.__class__.max_mass)))
    
    # -------------------------------------------------------------------------
    # Main force model
    
    def update_acceleration(self):
        '''
        Calculates main acceleration term from force-based model of environment.
        '''
        # Instantiate force term
        force_term = np.zeros(2)

        # Sum gravitational attractions
        for star in Star.iterate_class_instances():
            if star == self:
                continue
            # Gm1m2/(r^2) in direction towards other planet - note dist returns r^2
            force_term  += (Star.G*star.mass*self.mass) * self.unit_dirn(star)/(self.dist(star))
        
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
        new_dict["colour"] = self.colour
        return new_dict
    
    @classmethod
    def from_dict(cls, dict):
        instance = cls(position=np.array(dict["position"]),
                   velocity=np.array(dict["velocity"]),
                   unlinked=True)
        instance.acceleration = np.array(dict["acceleration"])
        instance.last_position = np.array(dict["last_position"])
        instance.mass = dict["mass"]
        instance.colour = dict["colour"]
        instance.alive = dict["alive"]
        return instance
    
    # -------------------------------------------------------------------------
    # Matplotlib

    def draw_plt(self, ax:plt.Axes, com=None, scale=None):
        '''
        Updates the stored self.plt_artist PathCollection with new position
        '''
        # Get plot position in frame
        plot_position = self.orient_to_com(com, scale)

        # Update artist with PathCollection.set_offsets setter method
        if self.plt_artists is None:
            # Initialise PathCollection artist as scatter plot point
            self.plt_artists = []
            self.plt_artists.append(ax.scatter(plot_position[0],plot_position[1], \
                                     s=self.size,c=[self.colour], cmap='gray', \
                                     vmin=0,vmax=1 ))
        else:
            # Update with offset
            self.plt_artists[0].set_offsets(plot_position)
        
        return self.plt_artists




