import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D

class Particle:
    '''
    Parent class for particles in a 2D plane
    '''
    # Manager
    manager = None

    # Default wall boundaries (Region is [0,env_x_lim]X[0,env_y_lim] )
    env_x_lim: float = 100
    env_y_lim: float = 100

    # Bools for tracking COM or torus points
    track_com: bool = False
    torus: bool = False
    
    # Default timestep
    DEFAULT_TIMESTEP: float = 0.01
    
    # Initialisation function
    def __init__(self,
                position: np.ndarray = None,
                velocity: np.ndarray = None,
                unlinked = False) -> None:
        '''
        Initiate generic particle, with optional position and velocity inputs.
        Start with random position, and zero velocity and zero acceleration.
        Set an ID value and then increment dictionaries
        '''
        # ---- ID ----

        # Initialise class name in state dict if not there
        if self.__class__.__name__ not in self.manager.state["Particle"].keys():
            self.manager.state["Particle"][self.__class__.__name__] = {}

        # Indexing for this instance
        if unlinked:
            self.id: int = -1
        else:
            self._initialise_instance_id()

        # Start alive
        self.alive: bool = True

        # ---- Motion ----

        # If no starting position given, assign random within 2D wall limits
        if position is None:
            self.position: np.ndarray = np.array([np.random.rand(1)[0]*(self.env_x_lim),np.random.rand(1)[0]*self.env_y_lim])
        else:
            self.position: np.ndarray = position

        # If no starting velocity given, set it to zero.
        if velocity is None:
            self.velocity: np.ndarray = np.zeros(2)
        else:
            self.velocity: np.ndarray = velocity

        # Set last position via backtracking
        self.last_position: np.ndarray = self.position - self.velocity*self.manager.delta_t

        # Initialise acceleration as zero, mass as 1
        self.acceleration: np.ndarray = np.zeros(2)
        self.mass: float = 1

        # Bools for external corrections outside of force model
        self.max_speed: float = None
        self.just_reflected: bool = False

        # Matplotlib artists
        self.plt_artists: list = None

    def _initialise_instance_id(self):
        # Get Child class name of current instance
        class_name = self.__class__.__name__
        
        # Get unused ID
        self.id: int = self.manager.max_ids_dict.get(class_name, -1) + 1

        # Update max_ids_dict
        if class_name not in self.manager.max_ids_dict.keys():
            self.manager.max_ids_dict[class_name] = self.id
        elif self.manager.max_ids_dict[class_name] < self.id:
            self.manager.max_ids_dict[class_name] = self.id

        # Add to state dict
        self.manager.state["Particle"][class_name][self.id] = self

    # -------------------------------------------------------------------------
    # Printing

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
    # Instance management utilities

    @classmethod
    def get_instance_by_id(cls, id):
        ''' Get class instance by its id. If id doesn't exist, throw a KeyError.'''
        existing_class_ids = cls.manager.state["Particle"].get(cls.__name__, {}).keys()
        if id in existing_class_ids:
            return cls.manager.state["Particle"][cls.__name__][id]
        else:
            raise KeyError(f"Instance with id {id} not found in {cls.__name__}.")
        
    def unalive(self):
        ''' Sets the class instance with this id to be not alive. '''
        self.alive = False

    @classmethod
    def iterate_class_instances(cls):
        """
        Generator to yield all particles of a certain child class.

        Yields:
            cls(Particle): All instances of cls
        """
        class_dict = cls.manager.state["Particle"].get(cls.__name__, {})
        for particle in list(class_dict.values()):
            if particle.alive:
                yield particle
    
    # -------------------------------------------------------------------------
    # Geometry
    
    '''
    Periodic boundaries -> We have to check different directions for shortest dist.
    Need to check tic-tac-toe grid of possible directions:
            x | x | x
            ---------
            x | o | x
            ---------
            x | x | x
    We work from top right, going clockwise.
    TODO: Use more sophisticated approach by mapping to unit circle?
    '''
    up, right = np.array([0,env_y_lim]), np.array([env_x_lim,0])
    torus_offsets = [np.zeros(2), up+right, right, -up+right, -up, -up-right, -right, up-right, up]

    def torus_dist(self,other):
        ''' Calculate distance, direction between particles in Toroidal space. '''
        directions = [(other.position + i) - self.position  for i in Particle.torus_offsets]
        distances = [np.sum(i**2) for i in directions]
        mindex = np.argmin(distances)
        return distances[mindex], directions[mindex]

    def dist(self,other, return_both: bool = False):
        ''' 
        Calculates SQUARED distance between particles.
        Works for regular Euclidean space as well as toroidal.
        If return_both then returns direction from self to other.
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
        Calculates the direction unit vector pointing from self to other.
        '''        
        dist, dirn = self.dist(other,return_both=True)
        return dirn/np.sqrt(dist)
               
    def enforce_speed_limit(self):
        ''' 
        Restrict magnitude of particle's velocity to its max speed.
        Backtracks current position if over speed limit.
        '''
        speed = np.sqrt(np.sum(self.velocity**2))
        if speed > self.max_speed:
            self.velocity *= self.max_speed/speed
            self.position = self.last_position + self.velocity*self.manager.delta_t

    def inelastic_collision(self, scale_factor:float=0.8):
        ''' 
        Reduce magnitude of particle's velocity by given scale factor.
        Backtracks current position.
        '''
        self.velocity *= scale_factor
        self.position = self.last_position + self.velocity*self.manager.delta_t
        self.just_reflected = False

    def torus_wrap(self):
        ''' Wrap particle position coordinates into toroidal space with modulo functions'''
        self.position = self.position % [Particle.env_x_lim, Particle.env_y_lim]

    @staticmethod
    def torus_1d_com(positions, masses, domain_length):
        """
        Get the centre of mass of an array of 1d positions and masses,
        over a 1D circular domain with domain_length

        - We map 1D domain (segment of R^1) to complex unit circle S^1 
        - We then average the new complex coords weighted by their masses
        - We finally map from complex coord's angle back to a point in the 1D domain

        Args:
            positions (np.array, 1d): Array of particle positions along 1 coordinate
            masses (np.array, 1d): Corresponding array of particle masses
            domain_length (float | int): The length of the coordinate domain (usually Particle.env_x_lim)

        Returns:
            float: Centre of mass along the 1D domain
        """
        # Convert positions to angles in [0,2pi]
        angles = positions * 2*np.pi/domain_length
        # Compute complex coordinates on unit circle S^1
        coords = np.exp(1j * angles)
        # Element-wise multiply mass by complex coord
        weighted_coords = np.multiply(masses, coords)
        # Sum complex coords over row axis, divide by total mass
        com = np.sum(weighted_coords, axis=0) / np.sum(masses)
        # Get com angle, map back to modulo [0,2pi]
        com_angle = np.angle(com) % (2*np.pi)
        # Map from angle back to 1D domain
        com_1d = com_angle * domain_length / (2*np.pi)
        return com_1d

    @staticmethod
    def centre_of_mass_calc(iterable):
        '''
        Calculate COM of objects in an iterable with 'mass' and 'position' attributes.
        Works for euclidean and toroidal spaces.
        '''
        # Get masses and coordinates as arrays
        masses = []
        positions = []
        for instance in iterable():
            masses.append(instance.mass)
            positions.append(instance.position) # np.array (2,)
        masses = np.array(masses)
        positions = np.array(positions) # np.array (iter_length, 2)

        # Compute COM based on space
        if Particle.torus:
            # We treat X and Y coords independently
            x_com = Particle.torus_1d_com(positions[:,0], masses, Particle.env_x_lim)
            y_com = Particle.torus_1d_com(positions[:,1], masses, Particle.env_y_lim)
            com = np.array([x_com,y_com])
        else:
            # Element-wise multiply mass by position
            weighted_positions = positions
            weighted_positions[:,0] = weighted_positions[:,0] * masses
            weighted_positions[:,1] = weighted_positions[:,1] * masses
            # Sum over row axis, divide by total mass
            com = np.sum(weighted_positions, axis=0) / np.sum(masses)
    
        return com

    @classmethod
    def centre_of_mass_class(cls):
        ''' Calculate COM of all class objects. '''
        return Particle.centre_of_mass_calc(cls.iterate_class_instances)
        
    @staticmethod
    def centre_of_mass():
        ''' Calculate COM of all alive Particle objects. '''
        return Particle.centre_of_mass_calc(Particle.manager.iterate_all_alive_particles)

    @staticmethod
    def scene_scale(com: np.ndarray):
        ''' Compute the maximum x or y distance a particle has from the COM. '''
        all_dists = []
        for instance in Particle.manager.iterate_all_alive_particles():
            all_dists.append((instance.position - com).tolist())
        return np.max(all_dists)
     
    def orient_to_com(self, com, scale):
        ''' Affine translation on point coordinates to prepare for plotting.  '''
        # Check both not None
        if com is None or scale is None:
            return self.position
        plot_centre = np.array([0.5*Particle.env_x_lim, 0.5*Particle.env_y_lim])
        minor_axis = np.min(plot_centre)
        # Change particle to COM frame, scale by dist to furthest particle, 
        # then translate origin to middle of plot
        return ((self.position - com) * 0.8*minor_axis/scale) + plot_centre
    
    @property
    def theta(self):
        ''' Particle's subtended angle from the x axis, then offset by 90 deg. '''
        return np.arctan2(self.velocity[1], self.velocity[0]) - np.pi/2

    # -------------------------------------------------------------------------
    # Main timestep function

    def update(self):
        '''
        Use Verlet Integration to update acceleration, velocity,
        position and last_position of Particle.

        x_next = 2*x_now - x_last + acc*(dt)^2
        Passes predicted new position through checks, including speed limits,
        and torus modulo function on coordinates, which may backtrack the new position.
        '''
        # Let particle update its acceleration via child class methods
        if self.update_acceleration():
            # particle no longer alive
            return

        # Find last position from velocity - avoids torus wrapping problems
        self.last_position = self.position - self.velocity*self.manager.delta_t

        # Verlet Integration
        # Use tuple unpacking so we dont need a temp variable
        self.position, self.last_position = 2*self.position - self.last_position +self.acceleration*((self.manager.delta_t)**2), self.position
        
        # Update velocity
        displacement = (self.position - self.last_position)
        self.velocity = displacement/self.manager.delta_t

        # Enforce speed limit
        if self.max_speed is not None:
            self.enforce_speed_limit()

        # Reduce speed after inelastic collision
        if self.just_reflected:
            self.inelastic_collision()

        # Enforce torus wrapping
        if Particle.torus:
            self.torus_wrap()
    
    # -------------------------------------------------------------------------
    # Logging

    def copy_state(self, new_object):
        ''' Copy core attributes from one particle to the other'''
        self.position = new_object.position
        self.velocity = new_object.velocity
        self.acceleration = new_object.acceleration
        self.last_position = new_object.last_position
        self.mass = self.mass
        self.alive = new_object.alive

    def to_dict(self):
        ''' Compose dictionary of core particle attributes '''
        new_dict = {
            "position":self.position.tolist(),
            "last_position":self.last_position.tolist(),
            "velocity":self.velocity.tolist(),
            "acceleration":self.acceleration.tolist(),
            "mass":self.mass,
            "alive":self.alive
        }
        return new_dict
    
    # -------------------------------------------------------------------------
    # Matplotlib

    def remove_from_plot_plt(self):
        ''' Remove the particle's matplotlib artists from the axes they're registered to. '''
        # Use matplotlib .remove() method which works on all artists
        try:
            for artist in self.plt_artists:
                artist.remove()
        except Exception as e:
            pass
        # Reset artists as None:
        # Next loop, plt_artists will be reinitialised from None inside plot
        self.plt_artists = None
        return []

    @staticmethod
    def create_triangle_plt(angle_rad):
        ''' Create irregular triangle marker for plotting instances. '''
        # Define vertices for an irregular triangle (relative to origin)
        triangle = np.array([[-0.5, -1], [0.5, -1], [0.0, 1]])
        # Create a rotation matrix
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad),  np.cos(angle_rad)]])
        # Apply the rotation to the triangle vertices
        return triangle @ rotation_matrix.T
    
    def plot_as_triangle_plt(self, ax, com=None, scale=None, facecolor='white', plot_scale=1):
        ''' Plot particle with directional triangle. '''
        # Remove artist from plot if dead
        if not self.alive:
            return self.remove_from_plot_plt()
        
        # Get plot position in frame
        plot_position = self.orient_to_com(com, scale)

        # Update artist with PathCollection.set_offsets setter method
        if self.plt_artists is None:
            # Create a Polygon patch to represent the irregular triangle
            triangle_shape = self.create_triangle_plt(self.theta)
            polygon = Polygon(triangle_shape, closed=True, facecolor=facecolor, edgecolor='black')
            # Create and apply transformation of the polygon to the point
            t = Affine2D().scale(plot_scale).translate(plot_position[0], plot_position[1]) + ax.transData
            polygon.set_transform(t)

            # Add to artists and axes
            self.plt_artists = [polygon]
            ax.add_patch(polygon)
        else:
            # Recompute orientation
            triangle_shape = self.create_triangle_plt(self.theta)

            # Update shape and transform
            self.plt_artists[0].set_xy(triangle_shape)
            t = Affine2D().scale(plot_scale).translate(self.position[0], self.position[1]) + ax.transData
            self.plt_artists[0].set_transform(t)
                
        return self.plt_artists