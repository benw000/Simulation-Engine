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
    # This is used to fully encode the system's state at each timestep.
    all = {}

    # Current time, to be updated each timestep
    current_time = 0
    num_timesteps = 100

    # Basic wall boundaries (Region is [0,walls_x_lim]X[0,walls_y_lim] )
    walls_x_lim = 100
    walls_y_lim = 100
    torus = False

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

    # -------------------------------------------------------------------------
    # Management utilities
    # TODO: Make some of these hidden!

    @classmethod
    def get_count(cls):
        ''' Return a class type count. eg  num_birds = Bird.get_count(). '''
        return Particle.pop_counts_dict.get(cls.__name__, 0)
    @classmethod
    def get_max_id(cls):
        ''' Return a class max id. eg max_id_birds = Bird.get_max_id(). '''
        return Particle.max_ids_dict.get(cls.__name__, 0)
    
    @classmethod
    def remove_by_id(cls, id):
        ''' Remove class instance from list of instances by its id. '''
        if id in Particle.all[cls.__name__]:
            del Particle.all[cls.__name__][id]
            Particle.pop_counts_dict[cls.__name__] -= 1
        else:
            pass

    @classmethod
    def get_instance_by_id(cls, id):
        ''' Get class instance by its id. If id doesn't exist, throw a KeyError.'''
        if id in Particle.all[cls.__name__]:
            return Particle.all[cls.__name__][id]
        else:
            raise KeyError(f"Instance with id {id} not found in {cls.__name__}.")
        
    @classmethod
    def iterate_instances(cls):
        ''' Iterate over all instances of a given class by id '''
        # This function is a 'generator' object in Python due to the use of 'yield'.
        # It unpacks each {id: instance} dictionary item within our Particle.all[classname] dictionary
        # It then 'yields' the instance. Can be used in a for loop as iterator.
        for id, instance in Particle.all.get(cls.__name__, {}).items():
            yield instance
        

    def __str__(self) -> str:
        ''' Print statement for particles. '''
        return f"Particle {self.id} at position {self.position} with velocity {self.velocity}."

    def __repr__(self) -> str:
        ''' Debug statement for particles. '''
        return f"{self.__class__.__name__}({self.id},{self.position},{self.velocity})"

    # -------------------------------------------------------------------------
    # Numerical utilities
    # TODO: Make some of these hidden!

    def dist(self,other) -> float:
        ''' 
        Calculates squared euclidean distance between particles.
        Usage: my_dist = particle1.dist(particle2).
        '''
        # TODO: optional bool for Torus shaped region
        return np.sum((self.position-other.position)**2)
    
    def dirn(self,other) -> float:
        '''
        Calculates the direction unit vector pointing from particle1 to particle2.
        Usage: my_vec = particle1.dist(particle2).
        '''
        # TODO: optional bool for Torus shaped region
        return (other.position-self.position)/np.sqrt(self.dist(other))
    
    def normalise_velocity(self, max_speed: float):
        ''' Hardcode normalise a particle's velocity to a specified max speed. '''
        speed = np.sqrt(np.sum(self.velocity)**2)
        if speed > max_speed:
            self.velocity *= max_speed/speed

    @staticmethod
    def centre_of_mass():
        # TODO: return COM, worked out from Particle.all instance.position and instance.mass
        pass

    @staticmethod
    def scene_bounds():
        # TODO: return furthest points from COM to bounds, get a scaling factor
        pass

    @staticmethod
    def timestep_update(speed_limit: bool = False):
        # Build class list by checking class.get_count() method
        # For each class in list, use iterator method to access each id
        # Use current state to work out acceleration for every existing particle
        # for particle in [iterator]:
        #       particle.update_acceleration()
        # For each particle update its (current, last) by Verlet recurrence relation using its acceleration
        # Implement speed limit by moving in direction of new (current-last) but only distance=particle.max_speed*delta_t along
        # If torus, pass final position through modulo functions
        # Any other main things to consider?
        pass
    

    # -------------------------------------------------------------------------
    # CSV utilities
    # Need to be smart to check variable lengths
    # TODO: Make some of these hidden!


    @staticmethod
    def write_to_csv(filename):
        # Flatten all info from current state Particle.all using heuristic
        # Seperate by Child class with designated character like * or -
        # Write to single row of csv for that timestep
        # As entities may die or reproduce, columns must be dynamic - so no header row
        # So need custom parsing
        # Timestep, time, class, num_class, pos_x_id, pos_y_id, ..., class, num_class, ...
        # (Need filename as datetime will change from creation)
        pass

    @staticmethod
    def load_from_csv(filename, timestep):
        # Parse particular row of CSV
        # Read * and classname following, restart indexing, read all until * character
        # Replace current Particle.all with what is read
        pass


    # -------------------------------------------------------------------------
    # Animation utilities
    # TODO: Make some of these hidden!

    @staticmethod
    def animate_timestep(timestep):
        # Nice print animation in progress - Can we take secondary arguments like total num timesteps?
        # would allow us to print progress. If not then need Particle.num_timesteps set manually
        # Call on load_from_csv ? to load in current Particle.all system state
        # This function will be called by FuncAnimation
        # Establish fig, ax
        # IF COM tracking: 
        #       Call on centre_of_mass, and scene_bounds
        # Iterate through classes with nonzero count:
        #       COM scale background elements positions
        #       cls.background_plot()   (Draw background like attractions, doors etc)
        # Iterate through every particle instance:
        #       COM scale particle positions
        #       particle.plot()   (Draw each particle according to its specific plot function)
        # plt.show() 
        pass

    


class Prey(Particle):
    '''
    Prey particle for flock of birds simulation.
    '''
    max_speed = 5

    mass = 7 

    def __init__(self, position: np.ndarray = None, velocity: np.ndarray = None) -> None:
        super().__init__(position, velocity)
        pass

    def update_acceleration(self):
        # go through all in Particle.all[Prey] and Particle.all[Predator] to work out forces
        # acceleration by dividing by self.mass
        # Use this point to track killing as well: use remove_by_id method if any Predator too close
        # Could have rule that this spawns in a new predator?? cool
        pass

    # -------------------------------------------------------------------------
    # Animation utilities

    def plot(self, fig, ax):
        # Plot individual Prey particle onto existing fig, ax
        # unpack fig, ax
        # ax.plot(self.position[0], self.position[1])   sort of thing
        pass

    @classmethod
    def background_plot(cls, fig, ax):
        # Similar sort of thing, using cls.wall_limit_x etc to draw things

        pass

