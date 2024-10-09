# classes.py - generalising pedestrian.py

import numpy as np


class Particle:
    '''
    Parent class for particles in a 2D plane
    '''
    # -------------------------------------------------------------------------
    # Attributes

    #  Dictionaries for population count and maximum ID number of each child class
    pop_counts_dict = {} 
    max_ids_dict = {}

    # Dictionary of all child classes, referenced by ID number
    # eg {bird: {0:instance0, 1:instance1, ...}, plane: {0: ...}, ... }
    # This is used to fully encode the system's state at each timestep.
    all = {}

    # Default time step length used for simulation
    delta_t = 0.01
    # Current time, to be updated each timestep
    current_time = 0
    num_timesteps = 100

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
        # ---------------
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

        # Initialise acceleration as attribute
        self.acceleration = np.zeros(2)

        # --------------
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
    # Instance management utilities
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
    def iterate_class_instances(cls):
        ''' Iterate over all instances of a given class by id. '''
        # This function is a 'generator' object in Python due to the use of 'yield'.
        # It unpacks each {id: instance} dictionary item within our Particle.all[classname] dictionary
        # It then 'yields' the instance. Can be used in a for loop as iterator.
        for id, instance in Particle.all.get(cls.__name__, {}).items():
            yield instance

    @staticmethod
    def iterate_all_instances():
        ''' Iterate over all existing child instances. '''
        # Create dictionary with all child instances
        dict_list = {}
        for i in Particle.all.values():
            dict_list.update(i)
        # Create generator through the dictionary values (instances)
        for id, instance in dict_list.items():
            yield instance
        
    def __str__(self) -> str:
        ''' Print statement for particles. '''
        return f"Particle {self.id} at position {self.position} with velocity {self.velocity}."

    def __repr__(self) -> str:
        ''' Debug statement for particles. '''
        return f"{self.__class__.__name__}({self.id},{self.position},{self.velocity})"

    # -------------------------------------------------------------------------
    # Distance utilities
    # TODO: Make some of these hidden!

    torus = False
    '''
    Periodic boundaries -> We have to check different directions for shortest dist.
    Need to check tic-tac-toe grid of possible directions:
            x | x | x
            ---------
            x | o | x
            ---------
            x | x | x
    We work from top right, going clockwise.
    '''
    up, right = np.array([0,walls_y_lim]), np.array([walls_x_lim,0])
    torus_offsets = [np.zeros(2), up+right, right, -up+right, -up, -up-right, -right, up-right, up]

    def torus_dist(self,other):
        directions = [(other.position + i) - self.position  for i in Particle.torus_offsets]
        distances = [np.sum(i**2) for i in directions]
        mindex = np.argmin(distances)
        return distances[mindex], directions[mindex]

    def dist(self,other, return_both: bool = False):
        ''' 
        Calculates SQUARED euclidean distance between particles.
        Usage: my_dist = particle1.dist(particle2).
        If Particle.torus, then finds shortest squared distance from set of paths.
        '''
        if Particle.torus:
            dist, dirn = self.torus_dist(other)
        else:
            dirn = other.position - self.position
            dist = np.sum((dirn)**2)

        if return_both:
            return dist, dirn
        else:
            return dist
            
    def unit_dirn(self,other) -> float:
        '''
        Calculates the direction unit vector pointing from particle1 to particle2.
        Usage: my_vec = particle1.dirn(particle2).
        '''        
        dist, dirn = self.dist(other,return_both=True)
        return dirn/np.sqrt(dist)
               
    def enforce_speed_limit(self):
        ''' Hardcode normalise a particle's velocity to a specified max speed. '''
        # Speed limit
        if self.max_speed is None:
            pass
        else:
            # Hardcode speed limit, restrict displacement
            speed = np.sqrt(np.sum(self.velocity**2))
            if speed > self.max_speed:
                # Change velocity
                self.velocity *= self.max_speed/speed
                # Change current position to backtrack
                self.position = self.last_position + self.velocity*Particle.delta_t

    @staticmethod
    def centre_of_mass():
        ''' Compute COM of all particle instances. '''
        total_mass = 0
        com = np.zeros(2)
        # Call generator to run over all particle instances
        for instance in Particle.iterate_all_instances:
            mass = instance.mass
            com += mass*instance.position
            total_mass += mass
        com *= 1/total_mass

    @staticmethod
    def scene_scale():
        ''' Compute the maximum x or y distance a particle has from the COM. '''
        com = Particle.centre_of_mass()
        max_dist = 0
        # Call generator to find max dist from COM
        for instance in Particle.iterate_all_instances:
            vec_from_com = instance.position - com
            for i in vec_from_com:
                if i > max_dist:
                    max_dist = i
                else:
                    pass
        return max_dist
        
    # -------------------------------------------------------------------------
    # Main timestep function

    @staticmethod
    def timestep_update():
        '''
        Main timestep function. 
        - Calls each child class instance to update its acceleration,
            according to its own force rules. 
        - Uses 'Verlet Integration' timestepping method, predicting instance's position after a 
            timestep using its current position, last position, and acceleration:
            x_next = 2*x_now - x_last + acc*(dt)^2
        - Passes predicted new position through checks, including speed limits,
            and torus modulo function on coordinates.
        '''
        for i in Particle.iterate_all_instances:
            # Let particle update its acceleration 
            i.update_acceleration()

            # Verlet Integration
            # Use tuple unpacking so we dont need a temp variable
            i.position, i.last_position = (2*i.position - i.last_position + \
                                            i.acceleration*(Particle.delta_t)**2), i.position
            
            # Update velocity
            displacement = (i.position - i.last_position)
            i.velocity = displacement/Particle.delta_t

            # Speed limit (if given)
            i.enforce_speed_limit()
            

            

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
