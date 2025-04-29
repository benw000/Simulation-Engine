import copy
import numpy as np
from scipy.stats import loguniform

from .parents import Particle



class Star(Particle):
    '''
    Star particle for N-body simulation
    '''
    # -------------------------------------------------------------------------
    # Attributes

    G = 1000
    min_mass = 10 
    max_mass = 10**3

    random_force = 0
    
    # Initialisation
    def __init__(self, position: np.ndarray = None, velocity: np.ndarray = None, id=None) -> None:
        '''
        Initialises a star object, inheriting from the Particle class.
        '''
        super().__init__(position, velocity, id)
        
        # Get mass from a log uniform distribution betwen min and max mass supplied
        self.mass = loguniform.rvs(Star.min_mass, Star.max_mass, size=1)[0]
        # Get velocity from 1/mass * 10 * random direction
        self.velocity = 10*np.array([np.random.rand(1)[0]*2 - 1,np.random.rand(1)[0]*2 - 1])

        # Random gray colour for plotting between 0.5 and 1
        self.colour = np.random.rand()/2 + 0.5

        # Ensure prototype for child class exists, callable by its name as a string only
        prototype = copy.copy(self)
        Particle.prototypes[self.__class__.__name__] = prototype

    def create_instance(self, id):
        ''' Used to create instance of the same class as self, without referencing class. '''
        return Star(id=id)

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
    # CSV utilities

    def write_csv_list(self):
        '''
        Format for compressing each Star instance into CSV.
        '''
        # Individual child instance info
        return [self.id, self.mass, self.colour, \
                self.position[0], self.position[1], \
                self.last_position[0],self.last_position[1],
                self.velocity[0], self.velocity[1],
                self.acceleration[0], self.acceleration[1] ]

    def read_csv_list(self, system_state_list, idx_shift):
        '''
        Format for parsing the compressed Star instances from CSV.
        '''
        self.mass = float(system_state_list[idx_shift+1])
        self.colour = float(system_state_list[idx_shift+2])
        self.position = np.array([float(system_state_list[idx_shift+3]), \
                                    float(system_state_list[idx_shift+4])])
        self.last_position = np.array([float(system_state_list[idx_shift+5]), \
                                    float(system_state_list[idx_shift+6])])
        self.velocity = np.array([float(system_state_list[idx_shift+7]), \
                                    float(system_state_list[idx_shift+8])])
        self.acceleration = np.array([float(system_state_list[idx_shift+9]), \
                                    float(system_state_list[idx_shift+10])])
        # Update idx shift to next id and return
        return idx_shift+11
    
     # -------------------------------------------------------------------------
    # Animation utilities

    def instance_plot(self, ax, com=None, scale=None):
        ''' 
        Plots individual Star particle onto existing axis. 
        '''

        # Get plot position in frame
        plot_position = self.position
        #size = 2*(2*np.log10(self.mass)+1)**3
        size = 5 + 10 * (np.power(2,np.log10(self.mass))-1)
        if (com is not None) and (scale is not None):
            plot_position = self.orient_to_com(com, scale)
            #size *= 1/np.sqrt(scale)
        #ax.scatter(plot_position[0], plot_position[1],marker='o',c=[self.colour], cmap='gray')
        
        ax.scatter(plot_position[0],plot_position[1],s=size,c=[self.colour], cmap='gray',vmin=0,vmax=1 )

    