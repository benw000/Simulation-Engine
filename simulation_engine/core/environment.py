import numpy as np
import matplotlib.pyplot as plt
from simulation_engine.core.particle import Particle

class Environment:
    '''
    Parent class for environmental obstacles in 2D plane
    '''
    # Attributes
    manager = None

    def __init__(self, unlinked=False):
        ''' Initialise an Environment object and add to manager's store. '''
        # Add self to manager
        class_name = self.__class__.__name__
        if not unlinked:
            if class_name in self.manager.state["Environment"]:
                self.manager.state["Environment"][class_name].append(self)
            else:
                self.manager.state["Environment"][class_name] = [self]
        # Initialise artists
        self.plt_artists = None

    # -------------------------------------------------------------------------
    # Logging

    def to_dict(self):
        ''' Compose dictionary of core particle attributes. (Currently empty) '''
        new_dict = {
        }
        return new_dict
    
    # -------------------------------------------------------------------------
    # Matplotlib

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

class Wall(Environment):
    '''
    Wall object characterised by directed line segment between two 2D points.
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
        ''' Compose dictionary of core attributes '''
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
        ''' Create Wall instance from core attributes in dictionary. '''
        instance = cls(a_position=np.array(new_dict["a_position"]),
                   b_position=np.array(new_dict["b_position"]),
                   line_colour=new_dict["line_colour"],
                   edge_colour=new_dict["edge_colour"],
                   unlinked=True)
        return instance 
    
    def copy_state(self, new_object):
        ''' Copy core attributes from one Wall to the other'''
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
        ''' Return Target object closest to a given particle. '''
        closest_target, closest_dist = None, np.inf
        for target in cls.manager.state["Environment"]["Target"]:
            dist = particle.dist(target)
            if dist < closest_dist:
                closest_dist = dist
                closest_target = target
        return closest_target

    # -------------------------------------------------------------------------
    # Logging

    def to_dict(self):
        ''' Compose dictionary of core attributes '''
        parent_dict = super().to_dict()
        new_dict = {
            "position":self.position.tolist(),
            "capture_thresh":self.capture_thresh,
            "colour":self.colour
        }
        return parent_dict | new_dict
    
    @classmethod
    def from_dict(cls, new_dict):
        ''' Create Target instance from core attributes in dictionary. '''
        instance = cls(position=np.array(new_dict["position"]),
                   colour=new_dict["colour"],
                   unlinked=True)
        instance.capture_thresh=new_dict["capture_thresh"]
        return instance
    
    def copy_state(self, new_object):
        ''' Copy core attributes from one Target to the other'''
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