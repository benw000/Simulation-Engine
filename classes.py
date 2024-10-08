# classes.py - generalising pedestrian.py

import numpy as np


class Particle:
    '''
    Parent class for particles in a 2D plane
    '''
    # -------------------------------------------------------------------------
    # Attributes

    #  Establish dictionaries to track the population count and maximum ID number
    #  for each child class.  eg {bird: 5, plane: 3, ...}
    pop_counts_dict = {} 
    max_ids_dict = {}   
    # Establish big dictionary of all child instances, referenced by ID number
    # eg {bird: {0:instance0, 1:instance1, ...}, plane: {0: ...}, ... }
    all = {}

    # Basic wall boundaries (Region is [0,walls_x_lim]X[0,walls_y_lim] )
    walls_x_lim = 100
    walls_y_lim = 100

    # Default time step length used for simulation
    delta_t = 0.01

    # Initialisation function
    def __init__(self,
                position: np.ndarray = None,
                velocity: np.ndarray = None) -> None:
        '''
        Initiate generic particle, with optional position and velocity inputs.
        Start with random position, and zero velocity and zero acceleration.
        Set an ID value and then increment dictionaries
        '''
        # ---------------------------------------------
        # Motion

        # If no starting position given, assign random within 2D wall limits
        if position is None:
            self.position = np.array([np.random.rand(1)[0]*self.walls_x_lim,np.random.rand(1)[0]*self.walls_y_lim])
        # If no starting velocity given, set it to zero.
        if velocity is None:
            self.velocity = np.zeros(2)

        # Extrapolate last position from starting position and velocity
        if velocity is None:
            self.last_position = self.position
        else:
            # v = (current-last)/dt , so last = current - v*dt
            self.last_position = self.position - self.velocity*self.delta_t

        # Start with zero acceleration to initialise as attribute
        self.acceleration = np.zeros(2)

        # ---------------------------------------------
        # Indexing

        # Get Child class name of current instance
        class_name = self.__class__.__name__
        
        # Population count
        if class_name not in Particle.pop_counts_dict:
            Particle.pop_counts_dict[class_name] = 0
        Particle.pop_counts_dict[class_name] += 1

        # ID - index starts at 0
        if class_name not in Particle.max_ids_dict:
            Particle.max_ids_dict[class_name] = 0
        else:
            Particle.max_ids_dict[class_name] += 1
        self.id = Particle.max_ids_dict[class_name]

        # Add instance to 'all' dict
        if class_name not in Particle.all:
            Particle.all[class_name] = {}
        Particle.all[class_name][self.id] = self

        # ---------------------------------------------

    # Management utilities
    @classmethod
    def get_count(cls):
        ''' Return a class type count. eg  num_birds = Bird.get_count(). '''
        class_name = cls.__name__
        return Particle.pop_counts_dict.get(class_name, 0)
    @classmethod
    def get_max_id(cls):
        ''' Return a class max id. eg max_id_birds = Bird.get_max_id(). '''
        class_name = cls.__name__
        return Particle.max_ids_dict.get(class_name, 0)
    
    @classmethod
    def remove_by_id(cls, id):
        ''' Remove class instance from list of instances by its id. '''
        class_name = cls.__name__
        if id in Particle.all[class_name]:
            del Particle.all[class_name][id]
            Particle.pop_counts_dict[class_name] -= 1








    def __str__(self) -> str:
        ''' Print statement for particles. '''
        return f"Particle {self.id} at position {self.position} with velocity {self.velocity}"

    def __repr__(self) -> str:
        ''' Debug statement for particles. '''
        return f"{self.__class__.__name__}({self.id},{self.position},{self.velocity})"
    
    # Numerical utilities
    def dist(self,other) -> float:
        ''' 
        Calculates squared euclidean distance between particles.
        Usage: my_dist = particle1.dist(particle2).
        '''
        return np.sum((self.position-other.position)**2)
    
    def dirn(self,other) -> float:
        '''
        Calculates the direction unit vector pointing from particle1 to particle2.
        Usage: my_vec = particle1.dist(particle2).
        '''
        return (other.position-self.position)/np.sqrt(self.dist(other))
    
    def normalise_velocity(self, max_speed: float):
        ''' Hard normalise a particle's speed to a specified max speed. '''
        speed = np.sqrt(np.sum(self.velocity)**2)
        if speed > max_speed:
            self.velocity *= max_speed/speed

    # General Verlet Integration timestep
    
    


class Prey(Particle):
    '''
    Prey particle for flock of birds simulation.
    '''
    # List of all 
    all = []
    def __init__(self, position: np.ndarray = None, velocity: np.ndarray = None) -> None:
        super().__init__(position, velocity)

