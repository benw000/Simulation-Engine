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
        # Call generator to find max dist from COM
        all_dists = []
        for instance in Particle.manager.iterate_all_alive_particles():
            all_dists.append((instance.position - com).tolist())
        max_dist = np.max(all_dists)

        return max_dist
     
    def orient_to_com(self, com, scale):
        ''' Affine translation on point coordinates to prepare for plotting.  '''
        # Check both not None
        if com is None or scale is None:
            return self.position
        # Transform
        centre = np.array([0.5*Particle.env_x_lim, 0.5*Particle.env_y_lim])
        term = np.min(centre)
        return centre + (self.position - com) * 0.8 * term/scale #* 1/scale
    
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
    
    @property
    def theta(self):
        return np.arctan2(self.velocity[1], self.velocity[0]) - np.pi/2

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
        self.position = new_object.position
        self.velocity = new_object.velocity
        self.acceleration = new_object.acceleration
        self.last_position = new_object.last_position
        self.alive = new_object.alive

    def to_dict(self):
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
        '''
        Create irregular triangle marker for plotting instances.
        '''
        # Define vertices for an irregular triangle (relative to origin)
        triangle = np.array([[-0.5, -1], [0.5, -1], [0.0, 1]])
        # Create a rotation matrix
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad),  np.cos(angle_rad)]])
        # Apply the rotation to the triangle vertices
        return triangle @ rotation_matrix.T
    
    def plot_as_triangle_plt(self, ax, com=None, scale=None, facecolor='white', plot_scale=1):
        ''' Used by birds in Predator Prey simulation '''
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


class Environment:
    '''
    Class containing details about the simulation environment, walls etc
    '''
    manager = None
    def __init__(self, unlinked=False):
        # Add self to manager
        class_name = self.__class__.__name__
        if not unlinked:
            if class_name in self.manager.state["Environment"]:
                self.manager.state["Environment"][class_name].append(self)
            else:
                self.manager.state["Environment"][class_name] = [self]
        # Initialise artists
        self.plt_artists = None

    def to_dict(self):
        new_dict = {
        }
        return new_dict
    
    @staticmethod
    def orient_to_com(position:np.ndarray, com, scale):
        ''' Affine translation on point coordinates to prepare for plotting.  '''
        # Check both not None
        if com is None or scale is None:
            return position
        # Transform
        centre = np.array([0.5*Particle.env_x_lim, 0.5*Particle.env_y_lim])
        term = np.min(centre)
        return centre + (position - com) * 0.8 * term/scale #* 1/scale
    
    def remove_from_plot_plt(self):
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

class Wall(Environment):
    '''
    Encodes instance of a wall
    '''
    DEFAULT_LINE_COLOUR = 'k'
    DEFAULT_EDGE_COLOUR = 'r'
    def __init__(self, a_position, b_position, line_colour=DEFAULT_LINE_COLOUR, edge_colour=DEFAULT_EDGE_COLOUR, unlinked=False) -> None:
        super().__init__(unlinked)
        self.a_position = a_position
        self.b_position = b_position
        self.line_colour = line_colour
        self.edge_colour = edge_colour
    
    # -------------------------------------------------------------------------
    # Printing
    
    def __str__(self) -> str:
        return f"Wall_[{self.a_position}]_[{self.b_position}]."
    
    # -------------------------------------------------------------------------
    # Geometry
    
    @property
    def wall_vec(self):
        return self.b_position - self.a_position
    
    @property
    def wall_length(self):
        return np.linalg.norm(self.wall_vec)

    @property
    def perp_vec(self):
        # Get perpendicular vector with 90 degrees rotation anticlockwise
        rot = np.array([[0,-1],
                        [1, 0]])
        return np.matmul(rot, self.wall_vec)
    
    def dist_to_wall(self, particle_position):
        '''
        Function taking a wall and particle with position.
        Returns the particle's closest distance to the wall, and the vector
        pointing from wall to particle (direction of repulsion force).
        '''
        x = particle_position
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

    # -------------------------------------------------------------------------
    # Logging

    def to_dict(self):
        parent_dict = super().to_dict()
        new_dict = {
            "a_position":self.a_position.tolist(),
            "b_position":self.b_position.tolist(),
            "line_colour":self.line_colour,
            "edge_colour":self.edge_colour
        }
        return parent_dict | new_dict
    
    @classmethod
    def from_dict(cls, new_dict):
        instance = cls(a_position=np.array(new_dict["a_position"]),
                   b_position=np.array(new_dict["b_position"]),
                   line_colour=new_dict["line_colour"],
                   edge_colour=new_dict["edge_colour"],
                   unlinked=True)
        return instance 
    
    def copy_state(self, new_object):
        self.a_position = new_object.a_position
        self.b_position = new_object.b_position
        self.line_colour = new_object.line_colour
        self.edge_colour = new_object.edge_colour

    # -------------------------------------------------------------------------
    # Matplotlib

    def draw_plt(self, ax:plt.Axes, com=None, scale=None):
        '''
        Updates the stored self.plt_artist PathCollection with new position
        '''
        if self.plt_artists is None:
            # Get positions
            a_plot_position = self.orient_to_com(self.a_position, com, scale)
            b_plot_position = self.orient_to_com(self.b_position, com, scale)
            x_vals = np.array([a_plot_position[0], b_plot_position[0]])
            y_vals = np.array([a_plot_position[1], b_plot_position[1]])

            # Initialise PathCollection artist as scatter plot point
            self.plt_artists = []
            self.plt_artists.append(ax.plot(x_vals, y_vals, c=self.line_colour)[0])
            self.plt_artists.append(ax.scatter(x_vals,y_vals,s=20,c=self.edge_colour))
        else:
            pass
            # Currently assuming Walls don't move
            # Update with offset
            # self.plt_artists[0].set_data(x_vals, y_vals)
            # self.plt_artists[1].set_offsets(np.column_stack((x_vals, y_vals)))

        return self.plt_artists

class Target(Environment):
    '''
    Encodes instance of a target
    '''
    def __init__(self, position, capture_radius= 0.5, colour='g', unlinked=False) -> None:
        super().__init__(unlinked)
        self.position = position
        self.capture_thresh = capture_radius**2
        self.colour = colour

    # -------------------------------------------------------------------------
    # Printing

    def __str__(self) -> str:
        return f"Target_[{self.position}]_[{self.capture_thresh}]]."
    
    # -------------------------------------------------------------------------
    # Geometry

    @classmethod
    def find_closest_target(cls, particle):
        closest_target = None
        dist = 100*Particle.env_x_lim**2
        for target in cls.manager.state["Environment"]["Target"]:
            if np.linalg.norm(target.position - particle.position) < dist:
                dist = np.linalg.norm(target.position - particle.position) 
                closest_target = target
        return closest_target

    # -------------------------------------------------------------------------
    # Logging

    def to_dict(self):
        parent_dict = super().to_dict()
        new_dict = {
            "position":self.position.tolist(),
            "capture_thresh":self.capture_thresh,
            "colour":self.colour
        }
        return parent_dict | new_dict
    
    @classmethod
    def from_dict(cls, new_dict):
        instance = cls(position=np.array(new_dict["position"]),
                   colour=new_dict["colour"],
                   unlinked=True)
        instance.capture_thresh=new_dict["capture_thresh"]
        return instance
    
    def copy_state(self, new_object):
        self.position = new_object.position
        self.capture_thresh = new_object.capture_thresh
        self.colour = new_object.colour

    # -------------------------------------------------------------------------
    # Matplotlib

    def draw_plt(self, ax:plt.Axes, com=None, scale=None):
        '''
        Updates the stored self.plt_artist PathCollection with new position
        '''
        if self.plt_artists is None:
            # Get position
            plot_position = self.orient_to_com(self.position, com, scale)
            # Initialise PathCollection artist as scatter plot point
            self.plt_artists = []
            self.plt_artists.append(ax.scatter(plot_position[0], plot_position[1], s=20, c=self.colour, marker='x'))
        else:
            pass
            # Currently assuming Targets dont move
            # Update with offset
            # self.plt_artists[0].set_offsets(plot_position)

        return self.plt_artists
