import copy
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D

from .parents import Particle, Environment

class Prey(Particle):
    '''
    Prey particle for flock of birds simulation.
    '''
    # -------------------------------------------------------------------------
    # Attributes

    prey_dist_thresh = 5**2
    prey_repulsion_force = 50

    pred_detect_thresh = 50**2
    pred_repulsion_force = 150

    pred_kill_thresh = 1**2

    com_attraction_force = 150

    random_force = 30
    
    # Initialisation
    def __init__(self, position: np.ndarray = None, velocity: np.ndarray = None, id=None) -> None:
        '''
        Initialises a Prey bird, inheriting from the Particle class.
        '''
        super().__init__(position, velocity, id)

        # Prey specific attributes
        self.mass = 0.5
        self.max_speed = 20

        # Ensure prototype for child class exists, callable by its name as a string only
        prototype = copy.copy(self)
        Particle.prototypes[self.__class__.__name__] = prototype

    def create_instance(self, id):
        ''' Used to create instance of the same class as self, without referencing class. '''
        return Prey(id=id)

    # -------------------------------------------------------------------------
    # Distance utilities

    def find_closest_pred(self):
        '''
        Returns instance of nearest predator to prey (self).
        '''
        # Initialise shortest distance as really large number
        # TODO: make this rely on actual span, not hardcoded
        shortest_dist = (10**5)**2
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
        '''
        # If predator near enough to kill prey instance, unalive prey and skip
        closest_pred = self.find_closest_pred()
        if closest_pred is not None:
            if self.dist(closest_pred) < self.pred_kill_thresh:
                self.unalive()
                # print(Particle.all)
                return 1
        
        # Instantiate force term
        force_term = np.zeros(2)

        # Prey repulsion force - currently scales with 1/d
        for bird in Prey.iterate_class_instances():
            if bird == self:
                continue
            elif self.dist(bird) < Prey.prey_dist_thresh:
                force_term += - self.unit_dirn(bird)*(self.prey_repulsion_force/(np.sqrt(self.dist(bird))))
            else:
                continue

        # Predator repulsion force
        for bird in Predator.iterate_class_instances():
            if self.dist(bird) < Prey.pred_detect_thresh:
                force_term += - self.unit_dirn(bird)*(self.pred_repulsion_force/(np.sqrt(self.dist(bird))))
            else:
                continue

    
        # Attraction to COM of prey
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
    # CSV utilities

    def write_csv_list(self):
        '''
        Format for compressing each Prey instance into CSV.
        '''
        # Individual child instance info
        return [self.id, \
                self.position[0], self.position[1], \
                self.last_position[0],self.last_position[1],
                self.velocity[0], self.velocity[1],
                self.acceleration[0], self.acceleration[1] ]

    def read_csv_list(self, system_state_list, idx_shift):
        '''
        Format for parsing the compressed Prey instances from CSV.
        '''
        self.position = np.array([float(system_state_list[idx_shift+1]), \
                                    float(system_state_list[idx_shift+2])])
        self.last_position = np.array([float(system_state_list[idx_shift+3]), \
                                    float(system_state_list[idx_shift+4])])
        self.velocity = np.array([float(system_state_list[idx_shift+5]), \
                                    float(system_state_list[idx_shift+6])])
        self.acceleration = np.array([float(system_state_list[idx_shift+7]), \
                                    float(system_state_list[idx_shift+8])])
        # Update idx shift to next id and return
        return idx_shift+9

    # -------------------------------------------------------------------------
    # Animation utilities

    @staticmethod
    def create_irregular_triangle(angle_rad):
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

    def instance_plot(self, ax, com=None, scale=None):
        ''' 
        Plots individual Prey particle onto existing axis. 
        '''

        # Get plot position in frame
        plot_position = self.position
        if (com is not None) and (scale is not None):
            plot_position = self.orient_to_com(com, scale)

        # Get direction angle from velocity
        theta = np.arctan2(self.velocity[1], self.velocity[0]) - np.pi/2

        # Create a Polygon patch to represent the irregular triangle
        triangle_shape = Prey.create_irregular_triangle(theta)
        polygon = Polygon(triangle_shape, closed=True, facecolor='white', edgecolor='black')
        
        # Create and apply transformation of the polygon to the point
        t = Affine2D().translate(plot_position[0], plot_position[1]) + ax.transData
        polygon.set_transform(t)

        # Plot polygon
        ax.add_patch(polygon)
        




# ------------------------------------------------------------------------------------------------------------------------




class Predator(Particle):
    '''
    Predator particle for flock of birds simulation.
    '''
    # -------------------------------------------------------------------------
    # Attributes

    prediction = True

    prey_attraction_force = 100

    pred_repulsion_force = 200
    pred_dist_thresh = 10**2

    pred_kill_thresh = 1**2

    random_force = 5
    
    # Initialisation
    def __init__(self, position: np.ndarray = None, velocity: np.ndarray = None, id = None) -> None:
        '''
        Initialises a Predator bird, inheriting from the Particle class.
        '''
        super().__init__(position, velocity, id)

        # Prey specific attributes
        self.mass = 0.5
        self.max_speed = 30

        # Ensure prototype for child class exists, callable by its name as a string only
        prototype = copy.copy(self)
        Particle.prototypes[self.__class__.__name__] = prototype

    def create_instance(self, id):
        ''' Used to create instance of the same class as self, without referencing class. '''
        return Predator(id=id)

    # -------------------------------------------------------------------------
    # Utilities

    def find_closest_prey(self):
        '''
        Returns instance of nearest pray to predator (self).
        '''
        # Initialise shortest distance as really large number
        # TODO: make this rely on actual span, not hardcoded
        shortest_dist = (10**5)**2
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

        # If near enough to kill prey instance, set acc and vel to 0
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
            if bird == self:
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
                closest_bird.position = target_position + 5*Particle.delta_t*target_velocity
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
    # CSV utilities

    def write_csv_list(self):
        '''
        Format for compressing each Predator instance into CSV.
        '''
        # Individual child instance info
        return [self.id, \
                self.position[0], self.position[1], \
                self.last_position[0],self.last_position[1],
                self.velocity[0], self.velocity[1],
                self.acceleration[0], self.acceleration[1] ]

    def read_csv_list(self, system_state_list, idx_shift):
        '''
        Format for parsing the compressed Predator instances from CSV.
        '''
        self.position = np.array([float(system_state_list[idx_shift+1]), \
                                    float(system_state_list[idx_shift+2])])
        self.last_position = np.array([float(system_state_list[idx_shift+3]), \
                                    float(system_state_list[idx_shift+4])])
        self.velocity = np.array([float(system_state_list[idx_shift+5]), \
                                    float(system_state_list[idx_shift+6])])
        self.acceleration = np.array([float(system_state_list[idx_shift+7]), \
                                    float(system_state_list[idx_shift+8])])
        # Update idx shift to next id and return
        return idx_shift+9

    # -------------------------------------------------------------------------
    # Animation utilities

    @staticmethod
    def create_irregular_triangle(angle_rad):
        '''
        Create irregular triangle marker for plotting instances.
        '''
        # Define vertices for an irregular triangle (relative to origin)
        triangle = np.array([[-0.5, -1], [0.5, -1], [0.0, 1]])*10
        # Create a rotation matrix
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad),  np.cos(angle_rad)]])
        # Apply the rotation to the triangle vertices
        return triangle @ rotation_matrix.T

    def instance_plot(self, ax, com=None, scale=None):
        ''' 
        Plots individual Predator particle onto existing axis. 
        '''

        # Get plot position in frame
        plot_position = self.position
        if (com is not None) and (scale is not None):
            plot_position = self.orient_to_com(com, scale)

        # Get direction angle from velocity
        theta = np.arctan2(self.velocity[1], self.velocity[0]) - np.pi/2

        # Create a Polygon patch to represent the irregular triangle
        triangle_shape = Prey.create_irregular_triangle(theta)
        polygon = Polygon(triangle_shape, closed=True, facecolor='red', edgecolor='black')
        
        # Create and apply transformation of the polygon to the point
        t = Affine2D().scale(20)
        t = Affine2D().translate(plot_position[0], plot_position[1]) + ax.transData
        polygon.set_transform(t)

        # Plot polygon
        ax.add_patch(polygon)

