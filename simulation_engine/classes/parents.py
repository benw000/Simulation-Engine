import os
import csv
import numpy as np

class Particle:
    '''
    Parent class for particles in a 2D plane
    '''
    # -------------------------------------------------------------------------
    # Attributes

    # Manager
    manager = None

    # Default wall boundaries (Region is [0,env_x_lim]X[0,env_y_lim] )
    env_x_lim: float = 100
    env_y_lim: float = 100

    # Bools for tracking COM or torus points
    track_com: bool = False
    torus = False
    
    # Initialisation function
    def __init__(self,
                position: np.ndarray = None,
                velocity: np.ndarray = None,
                id = None) -> None:
        '''
        Initiate generic particle, with optional position and velocity inputs.
        Start with random position, and zero velocity and zero acceleration.
        Set an ID value and then increment dictionaries
        '''
        # ---- ID ----
        self.alive=1

        # Indexing for this instance
        self._initialize_instance(id)

        # TODO: Add self to state dictionary with newest ID
        self.manager.state["Particle"][self.__class__.__name__][self.id] = self

        # ---- Motion ----
        self.max_speed = None
        self.just_reflected = False

        # If no starting position given, assign random within 2D wall limits
        if position is None:
            self.position = np.array([np.random.rand(1)[0]*(self.env_x_lim),np.random.rand(1)[0]*self.env_y_lim])
        else:
            self.position = position

        # If no starting velocity given, set it to zero.
        if velocity is None:
            self.velocity = np.zeros(2)
            # Extract last position as being the same
            self.last_position = self.position
        else:
            self.velocity = velocity
            # v = (current-last)/dt , so last = current - v*dt
            self.last_position = self.position - self.velocity*self.delta_t

        # Initialise acceleration as attribute
        self.acceleration = np.zeros(2)

        

    def _initialize_instance(self,id):
        """Initialize instance-specific attributes."""
        # Get Child class name of current instance
        class_name = self.__class__.__name__
        
        # ID
        if id is not None:
            # Set input ID
            id = int(id)
            self.id = id
        else:            
            self.id = self.manager.max_ids_dict.get(class_name, -1) + 1
        # Update max_ids_dict
        if class_name not in self.manager.max_ids_dict.keys():
            self.manager.max_ids_dict[class_name] = self.id
        elif self.manager.max_ids_dict[class_name] < self.id:
            self.manager.max_ids_dict[class_name] = self.id

        # Initialise class name in state dict if not there
        if class_name not in self.manager.state["Particle"].keys():
            self.manager.state["Particle"][class_name] = {}
        self.manager.state["Particle"][class_name][self.id] = self

        

    # -------------------------------------------------------------------------
    # Instance management utilities
    # TODO: Make these hidden and move to Manager when possible

    @classmethod
    def get_count(cls):
        ''' Return a class type count. eg  num_birds = Bird.get_count(). '''
        return len(cls.manager.state["Particle"].get(cls.__name__, {}).keys())
    
    @classmethod
    def get_max_id(cls):
        ''' Return a class max id. eg max_id_birds = Bird.get_max_id(). '''
        return cls.manager.max_ids_dict.get(cls.__name__, 0)
    
    @classmethod
    def get_instance_by_id(cls, id):
        ''' Get class instance by its id. If id doesn't exist, throw a KeyError.'''
        existing_class_ids = cls.manager.state["Particle"].get(cls.__name__, {}).keys()
        if id in existing_class_ids:
            return cls.manager.state["Particle"][cls.__name__][id]
        else:
            raise KeyError(f"Instance with id {id} not found in {cls.__name__}.")
        
    def unalive(self):
        ''' 
        Sets the class instance with this id to be not alive, decrements the class count.
        '''
        self.alive=0
        # Remove from manager
        self.manager.state["Particle"][self.__class__.__name__].remove(self.id)

    @classmethod
    def iterate_class_instances(cls):
        ''' Iterate over all instances of a given class by id. '''
        # This function is a 'generator' object in Python due to the use of 'yield'.
        # It unpacks each {id: instance} dictionary item within our Particle.all[classname] dictionary
        # It then 'yields' the instance. Can be used in a for loop as iterator.
        for instance in cls.manager.state["Particle"].get(cls.__name__, {}).values():
            if instance.alive == 1:
                yield instance
    
    # -------------------------------------------------------------------------
    # Dunder

    def __str__(self) -> str:
        ''' Print statement for particles. '''
        if self.alive==1:
            return f"Particle {self.id} at position {self.position} with velocity {self.velocity}."
        else:
            return f"Dead Particle {self.id} at position {self.position} with velocity {self.velocity}."

    def __repr__(self) -> str:
        ''' Debug statement for particles. '''
        if self.alive==1:
            return f"{self.__class__.__name__}({self.id},{self.position},{self.velocity})"
        else:
            return f"dead_{self.__class__.__name__}({self.id},{self.position},{self.velocity})"

    # -------------------------------------------------------------------------
    # Distance utilities
    
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
    up, right = np.array([0,env_y_lim]), np.array([env_x_lim,0])
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
        # Hardcode speed limit, restrict displacement
        speed = np.sqrt(np.sum(self.velocity**2))
        if speed > self.max_speed:
            # Change velocity
            self.velocity *= self.max_speed/speed
            # Change current position to backtrack
            self.position = self.last_position + self.velocity*self.manager.delta_t

    def inelastic_collision(self):
        if self.just_reflected:
            self.velocity *= 0.8
            self.position = self.last_position + self.velocity*self.manager.delta_t
            self.just_reflected = False

    def torus_wrap(self):
        ''' Wrap coordinates into Torus world with modulo functions'''
        x,y = self.position
        x = x % Particle.env_x_lim
        y = y % Particle.env_y_lim
        self.position = np.array([x,y])

    @classmethod
    def centre_of_mass_class(cls):
        ''' Compute COM of all particle instances. '''
        total_mass = 0
        com = np.zeros(2)
        # Call generator to run over all particle instances
        for instance in cls.iterate_class_instances():
            mass = instance.mass
            com += mass*instance.position
            total_mass += mass
        com *= 1/total_mass
        return com
    
    @staticmethod
    def centre_of_mass():
        ''' Compute COM of all particle instances. '''
        total_mass = 0
        com = np.zeros(2)
        # Call generator to run over all particle instances
        for instance in Particle.manager.iterate_all_particles():
            mass = instance.mass
            com += mass*instance.position
            total_mass += mass
        com *= 1/total_mass
        return com

    @staticmethod
    def scene_scale():
        ''' Compute the maximum x or y distance a particle has from the COM. '''
        com = Particle.centre_of_mass()
        max_dist = 0.01
        # Call generator to find max dist from COM
        for instance in Particle.manager.iterate_all_particles():
            vec_from_com = instance.position - com
            for i in vec_from_com:
                if i > max_dist:
                    max_dist = i
                else:
                    pass
        return max_dist
     
    def orient_to_com(self, com, scale):
        ''' Affine translation on point coordinates to prepare for plotting.  '''
        # Check both not None
        if com is None or scale is None:
            return self.position
        # Transform
        centre = np.array([0.5*Particle.env_x_lim, 0.5*Particle.env_y_lim])
        term = np.min(centre)
        return centre + (self.position - com) *0.9*term/scale #* 1/scale
    
    def find_closest_target(self):
        # Check through list of targets
        closest_target = None
        closest_dist = 10**5
        for target in self.manager.state["Environment"]["Target"]:
            vec = target.position - self.position
            dist = np.linalg.norm(vec)
            if dist < closest_dist:
                closest_dist = dist
                closest_target = target
        return closest_target

    # -------------------------------------------------------------------------
    # Main timestep function

    def update(self):
        '''
        - Uses 'Verlet Integration' timestepping method, predicting instance's position after a 
            timestep using its current position, last position, and acceleration:
            x_next = 2*x_now - x_last + acc*(dt)^2
        - Passes predicted new position through checks, including speed limits,
            and torus modulo function on coordinates.
        '''
        # Let particle update its acceleration 
        flag = self.update_acceleration()
        if flag==1:
            # i has been killed
            return

        # Find last position from velocity - avoids torus wrapping problems
        self.last_position = self.position - self.velocity*self.delta_t

        # Verlet Integration
        # Use tuple unpacking so we dont need a temp variable
        self.position, self.last_position = 2*self.position - self.last_position +self.acceleration*((self.delta_t)**2), self.position
        
        # Update velocity
        displacement = (self.position - self.last_position)
        self.velocity = displacement/self.delta_t

        # Enforce speed limit
        if self.max_speed is not None:
            self.enforce_speed_limit()

        # Reduce speed after inelastic collision
        self.inelastic_collision()

        # Enforce torus wrapping
        if Particle.torus:
            self.torus_wrap()
    
    # -------------------------------------------------------------------------
    # CSV utilities

    @staticmethod
    def write_state_to_csv():
        '''
        Takes Particle system state at the current time, and compresses into CSV.
        Iterates through each class, and within that each class instance.
        Calls each class's own method to write its own section.
        '''
        #--------------------------------
        if not Particle.csv_path.endswith('.csv'):
            Particle.csv_path += '.csv'

        # Compose CSV row entry
        system_state_list = [Particle.current_step, Particle.current_time]

        # Iterate through all current child classes
        for classname in Particle.pop_counts_dict.keys():

            # Access the class directly from the all_instances dictionary
            class_instances = Particle.all.get(classname, {})

            # Initialise class specific list
            class_list = [classname, Particle.pop_counts_dict[classname]]

            # Iterate through all instances of the child class
            for child in class_instances.values():
                if child.alive == 1:
                    # Add instance info to list using its write_csv_list function
                    class_list += child.write_csv_list()

            # Add child class info to main list
            class_list += ['|']
            system_state_list += class_list

        # End CSV row with 'END'
        system_state_list += ['END']

        # ------------------------------------
        # Writing entry to file

        if not os.path.exists(Particle.csv_path):
            with open(Particle.csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                header_row = ['Timestep', 'Time', 'ClassName', 'ClassPop', 'InstanceID', 'Attributes', '...','|','ClassName','...','|','END']
                writer.writerow(header_row)
                writer.writerow(system_state_list)
        else:
            with open(Particle.csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(system_state_list)

    @staticmethod
    def load_state_from_csv(timestep):
        '''
        Reads from a CSV containing the compressed Particle system state at a specific time.
        Iterates through each class, and within that each class instance.
        Parses to decompress the format outlined in write_state_to_csv.
        '''
        # ------------------------------------
        # Read row from CSV

        with open(Particle.csv_path, mode='r', newline='') as file:
            # Loop through the CSV rows until reaching the desired row
            # (This must be done since CSV doesn't have indexed data structure)
            reader = csv.reader(file)
            target_row_index = timestep+1 
            for i, row in enumerate(reader):
                if i == target_row_index:
                    system_state_list = row.copy()
                    break
        
        # ------------------------------------
        # Parse row into a full Particle system state

        # Parse timestep info, shift index
        Particle.current_step, Particle.current_time = system_state_list[0], system_state_list[1]
        idx_shift = 2

        # Loop through blocks for each child class
        while True:
            # Check if reached the end of row
            if system_state_list[idx_shift] == 'END':
                break
            
            # Parse class and number of instances, shift index
            classname = system_state_list[idx_shift]
            class_pop = int(system_state_list[idx_shift+1])
            idx_shift += 2

            # Diagnostic print
            #print(f"Timestep {timestep}, class_pop {class_pop}, classname {classname}")

            # TODO: Note this structure is faulty - can't run if we start with 0 instances of a class
            if class_pop == 0:
                # Remove all instances from current system
                Particle.pop_counts_dict[classname] = 0
                Particle.max_ids_dict[classname] = -1
                Particle.all[classname] = {}
            else:
                '''
                # Search for existing instances
                existing_id = None
                for key, value in Particle.all[classname].items():
                    existing_id = key
                if existing_id is not None:
                    # We have a valid instance, clone it into prototype for future use
                    prototype = copy.copy(Particle.all[classname][existing_id])
                    Particle.prototypes[classname] = prototype
                '''
                # Cull everything and start again with prototype, rebuild with CSV info
                Particle.pop_counts_dict[classname] = 0
                Particle.max_ids_dict[classname] = -1
                Particle.all[classname] = {}
                prototype = Particle.prototypes[classname]

                # Create class_pop many clones by looping through CSV row
                for i in range(class_pop):
                    id = int(system_state_list[idx_shift])
                    # Clone our prototype with it's child class's create_instance method
                    child = prototype.create_instance(id=id)

                    # Assign attributes by reading the system_state_list for that class
                    # This calls to child class's method to read each instance
                    idx_shift = child.read_csv_list(system_state_list, idx_shift)
                    
                # Check for pipe | at the end, then move past it
                if system_state_list[idx_shift] != '|':
                    raise IndexError(f"Something wrong with parsing, ~ column {idx_shift}.")
            
            # Move on to next class by shifting over pipe character '|'
            idx_shift += 1
        # Diagnostics
        #print("-")
        #print("done loading")
        #print(Particle.all)











# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------







class Environment:
    '''
    Class containing details about the simulation environment, walls etc
    '''
    manager = None
    def __init__(self):
        # Add self to manager
        self.manager.state[self.__class__.__name__].append(self)
    # TODO: Fill in any shared things between Wall and Target etc

    

class Wall(Environment):
    '''
    Encodes instance of a wall
    '''
    def __init__(self, a_position, b_position, line_colour='k', edge_colour='r') -> None:
        super.__init__()
        self.a_position = a_position
        self.b_position = b_position
        self.wall_vec = b_position - a_position
        self.wall_length = np.sqrt(np.sum((self.wall_vec)**2))
        self.line_colour = line_colour
        self.edge_colour = edge_colour
        # Get perpendicular vector with 90 degrees rotation anticlockwise
        self.perp_vec = np.array([[0,-1],[1,0]]) @ self.wall_vec

    def __str__(self) -> str:
        return f"Wall_[{self.a_position}]_[{self.b_position}]."

    def instance_plot(self, ax):
        x_vals = np.array([self.a_position[0], self.b_position[0]])
        y_vals = np.array([self.a_position[1], self.b_position[1]])
        ax.plot(x_vals, y_vals, c=self.line_colour)
        ax.scatter(x_vals,y_vals,s=20,c=self.edge_colour)

    def dist_to_wall(self, particle: Particle):
        '''
        Function taking a wall and particle with position.
        Returns the particle's closest distance to the wall, and the vector
        pointing from wall to particle (direction of repulsion force).
        '''
        x = particle.position
        a = self.a_position
        b = self.b_position
        vec = self.wall_vec # b-a
        length = self.wall_length
        
        # Check distance to point A (pole A)
        tolerance = 1e-6
        ax = np.linalg.norm(a - x)
        if ax < tolerance:
            # Particle is effectively at pole A
            return ax, (a - x)
        
        # Check distance to point B (pole B)
        bx = np.linalg.norm(b - x)
        if bx < tolerance:
            # Particle is effectively at pole B
            return bx, (b - x)
        
        # Projection of vector from A to particle onto the wall vector
        t = np.dot((x - a), vec) / (length * length)

        # If t < 0, the particle is closer to pole A
        if t < 0:
            return ax, -(a - x)
        # If t > 1, the particle is closer to pole B
        if t > 1:
            return bx, -(b - x)
        
        # Else 0 <= t <= 1, and the particle is perpendicular to the wall
        projection = a + t * vec
        x_to_wall = projection - x
        return np.sqrt(np.sum(x_to_wall**2)), -x_to_wall
    

class Target(Environment):
    '''
    Encodes instance of a target
    '''
    def __init__(self, position, capture_radius= 0.5, colour='g') -> None:
        super().__init__()
        self.position = position
        self.capture_thresh = capture_radius**2
        self.colour = colour

    def __str__(self) -> str:
        return f"Target_[{self.position}]_[{self.capture_thresh}]]."

    def instance_plot(self, ax):
        ax.scatter(self.position[0],self.position[1],s=20, c=self.colour, marker='x')