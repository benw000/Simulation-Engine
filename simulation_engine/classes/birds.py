import numpy as np

from simulation_engine.utils.errors import SimulationEngineInputError
from simulation_engine.classes import Particle
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
    # Create manager instance
    manager = Manager(args = args, 
                      show_graph = False,
                      draw_backdrop_plt_func = draw_backdrop_plt)
    
    # Add Prey, Predator child classes to registry
    manager.class_objects_registry["Prey"] = Prey
    manager.class_objects_registry["Predator"] = Predator

    # Split by mode
    if args.mode == 'run':
        return setup_run(args, manager)
    elif args.mode == 'load':
        return manager

def setup_run(args, manager):
    # Validate args
    if not len(args.nums) == 2:
        raise SimulationEngineInputError("(-n, --nums) Please supply 2 arguments for population when using birds simulation type")
    
    # Set Particle geometry attributes
    Particle.env_x_lim = 100
    Particle.env_y_lim = 100
    Particle.track_com = False
    Particle.torus = True

    # Initialise particles - could hide this in Particle but nice to be explicit
    num_prey, num_predators = args.nums[0], args.nums[1]
    for i in range(num_prey):
        Prey()
    for i in range(num_predators):
        Predator()

    return manager

def draw_backdrop_plt(ax):
    """
    Get an ax from manager, and plot things on it related to this mode
    Overrides Manager.default_draw_backdrop_plt

    Args:
        ax (plt.Axes): Main matplotlib frame
    """
    # Black background
    ax.set_facecolor('skyblue')



class Prey(Particle):
    '''
    Prey particle for flock of birds simulation.
    '''
    # -------------------------------------------------------------------------
    # Force model constants

    # Prey-prey repulsion
    prey_dist_thresh = 10**2
    prey_repulsion_force = 50
    # Predator-prey repulsion
    pred_detect_thresh = 50**2
    pred_repulsion_force = 150
    # Predator kill thresh
    pred_kill_thresh = 2**2
    # Prey COM attraction
    com_attraction_force = 75
    # Random motion
    random_force = 50
    
    # Initialisation
    def __init__(self, position: np.ndarray = None, velocity: np.ndarray = None, unlinked=None) -> None:
        '''
        Initialises a Prey bird, inheriting from the Particle class.
        '''
        super().__init__(position, velocity, unlinked)

        # Prey specific attributes
        self.mass = 0.5
        self.max_speed = 20

    # -------------------------------------------------------------------------
    # Distance utilities

    def find_closest_pred(self):
        '''
        Returns instance of nearest predator to prey (self).
        '''
        # Initialise shortest distance as really large number
        shortest_dist = Particle.env_x_lim**2 + Particle.env_y_lim**2
        closest_bird = None
        for bird in Predator.iterate_class_instances():
            dist = self.dist(bird) # squared dist
            if dist < shortest_dist:
                shortest_dist = dist
                closest_bird = bird
        return closest_bird

    # -------------------------------------------------------------------------
    # Main force model 

    def update_acceleration(self):
        '''
        Calculates main acceleration term from force-based model of environment.

        1. If prey is near enough to closest predator, kill it and return early
        2. Prey repulsion force within radius - scales with 1/d^2
        3. Predator repulsion force within detection thresh - scales with 1/d
        '''
        # 1. If predator near enough to kill prey instance, unalive prey and skip
        closest_pred = self.find_closest_pred()
        if closest_pred is not None:
            if self.dist(closest_pred) < self.pred_kill_thresh:
                self.unalive()
                return 1
        
        # Instantiate force term
        force_term = np.zeros(2)

        # 2. Prey repulsion force - currently scales with 1/d^2
        for bird in Prey.iterate_class_instances():
            if bird is self:
                continue
            elif self.dist(bird) < self.prey_dist_thresh:
                force_term += - self.unit_dirn(bird)*(self.prey_repulsion_force/self.dist(bird))
            else:
                continue

        # 3. Predator repulsion force - currently scales with 1/d
        for bird in Predator.iterate_class_instances():
            if self.dist(bird) < self.pred_detect_thresh:
                force_term += - self.unit_dirn(bird)*(self.pred_repulsion_force/(np.sqrt(self.dist(bird))))
            else:
                continue

        # 4. Attraction to COM of prey
        com = Prey.centre_of_mass_class()
        attract_dist = np.sum((com - self.position)**2)
        force_term += (com - self.position)*(self.com_attraction_force/(attract_dist))

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
        instance.max_speed = dict["max_speed"]
        instance.alive = dict["alive"]

        return instance

    # -------------------------------------------------------------------------
    # Matplotlib
    
    def draw_plt(self, ax, com=None, scale=None):
        facecolor = 'white'
        plot_scale = 1.5
        return self.plot_as_triangle_plt(ax, com, scale, facecolor, plot_scale)
        

class Predator(Particle):
    '''
    Predator particle for flock of birds simulation.
    '''
    # -------------------------------------------------------------------------
    # Force model
    
    # Attraction to prey
    prey_attraction_force = 100
    # Predator-Predator repulsion
    pred_repulsion_force = 100
    pred_dist_thresh = 100**2
    # Prey kill threshold
    pred_kill_thresh = 1**2
    # Random motion
    random_force = 5

    # Aim in the direction of extrapolated velocity
    prediction = True
    
    # Initialisation
    def __init__(self, position: np.ndarray = None, velocity: np.ndarray = None, unlinked=False) -> None:
        '''
        Initialises a Predator bird, inheriting from the Particle class.
        '''
        super().__init__(position, velocity, unlinked)

        # Predator specific attributes
        self.mass = 0.5
        self.max_speed = 30

    # -------------------------------------------------------------------------
    # Geometry

    def find_closest_prey(self):
        '''
        Returns instance of nearest pray to predator (self).
        '''
        # Initialise shortest distance as really large number
        shortest_dist = Particle.env_x_lim**2 + Particle.env_y_lim**2
        closest_bird = None
        for bird in Prey.iterate_class_instances():
            dist = self.dist(bird) # squared dist
            if dist < shortest_dist:
                shortest_dist = dist
                closest_bird = bird
        return closest_bird

    # -------------------------------------------------------------------------
    # Main force model 

    def update_acceleration(self):
        '''
        Calculates main acceleration term from force-based model of environment.
        '''

        # If near enough to kill prey instance, set acc to 0 and vel to low
        closest_bird = self.find_closest_prey()
        if closest_bird is not None:
            if self.dist(closest_bird) < self.pred_kill_thresh:
                closest_bird.unalive()
                self.acceleration = np.zeros(2)
                self.velocity *= 0.1
                return 0
        
        # Instantiate force term
        force_term = np.zeros(2)

        # Predator repulsion force - currently scales with 1/d
        for bird in Predator.iterate_class_instances():
            if bird is self:
                continue
            elif self.dist(bird) < Predator.pred_dist_thresh:
                force_term += - self.unit_dirn(bird)*(self.pred_repulsion_force/(np.sqrt(self.dist(bird))))
            else:
                continue

        # Attraction to closest prey
        if closest_bird is None:
            pass
        else:
            if Predator.prediction:
                target_position = closest_bird.position.copy()
                target_velocity = closest_bird.velocity
                # Temporarily change closest bird's position
                closest_bird.position = target_position + 5*self.manager.delta_t*target_velocity
                # Increment force
                force_term += self.unit_dirn(closest_bird)*(self.prey_attraction_force)
                # Change closest's bird position back
                closest_bird.position = target_position
            else:
                force_term += self.unit_dirn(closest_bird)*(self.prey_attraction_force)

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
        new_dict["mass"] = self.mass
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
        instance.max_speed = dict["max_speed"]
        return instance

    # -------------------------------------------------------------------------
    # Matplotlib

    def draw_plt(self, ax, com=None, scale=None):
        facecolor = 'red'
        plot_scale = 2
        return self.plot_as_triangle_plt(ax, com, scale, facecolor, plot_scale)