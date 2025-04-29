import copy
import numpy as np
from .parents import Particle

class Solid(Particle):
    '''
    Solid particle for the lattice of springs simulation
    '''
    # -------------------------------------------------------------------------
    # Attributes

    # Forces
    spring_length = 12 # use 12 or 10
    spring_constant = 3
    damping_constant = 2.5
    random_force = 0.01

    # Links
    links_count = 0
    links_dict = {}

    
    # Initialisation
    def __init__(self, position: np.ndarray = None, velocity: np.ndarray = None, id=None, loading=False) -> None:
        '''
        Initialises a solid object, inheriting from the Particle class.
        '''
        super().__init__(position, velocity,id)

        self.mass = 1

        self.connected_list = []
        if not loading:
            # Upon initialising new solid, check for points nearby
            # Loop through other existing solids
            for other in Solid.iterate_class_instances():
                if other is self:
                    pass
                else:
                    # Check distance is close enough to self
                    if self.dist(other) < Solid.spring_length * 1.2:
                        # Form link between self, other, check its not already in links_dict
                        link = [self.id, other.id]
                        link_is_new = True
                        for key, val in Solid.links_dict.items():
                            invert_val = [val[1],val[0]]
                            if link == val or link == invert_val:
                                link_is_new = False
                        if link_is_new:
                            # Update count, add to global dict
                            Solid.links_count += 1
                            Solid.links_dict[Solid.links_count] = link
                            # Update list of connected
                            self.connected_list += [other.id]
                            # Problem - when loading, new instance is created with random position but the correct id.
                            # It then falsely makes connections with other existing particles.
                            # It adds these connections to its own list, but also to the lists of existing particles
                            # which have already been loaded in and reread their connections.
                            # The loading stage rereads the connections, but nothing stops other random ghost prototype
                            # particles making false connections and smearing them on existing ones
                            other.connected_list += [self.id]
                        


        # Ensure prototype for child class exists, callable by its name as a string only
        prototype = copy.copy(self)
        Particle.prototypes[self.__class__.__name__] = prototype

    def create_instance(self,id):
        ''' Used to create instance of the same class as self, without referencing class. '''
        return Solid(id=id, loading=True)

    # -------------------------------------------------------------------------
    # Main force model
    
    def update_acceleration(self):
        '''
        Calculates main acceleration term from force-based model of environment.
        '''
        # On first timestep, if solid is not connected to others, kill it
        if Particle.current_step == 0 and self.connected_list == []:
            self.unalive()
            
        # Instantiate force term
        force_term = np.zeros(2)

        # Spring force from links
        '''
        for link_id, link in self.my_links.items():
            # Unpack link, other is the id that doesnt match self
            if link[0] == self.id:
                other = Solid.get_instance_by_id(link[1])
            elif link[1] == self.id:
                other = Solid.get_instance_by_id(link[0])
            else:
                raise ValueError(f"Link {link_id} ({link}) doesn't involve solid with id {self.id} despite being in its my_links dictionary.")
        '''
        for other_id in self.connected_list:
            other = Solid.get_instance_by_id(other_id)
            # Get dist^2 and direction to other
            dist, dirn = self.dist(other, return_both=True)
            # Spring force term based on extension
            extension = Solid.spring_length - np.sqrt(dist)
            force_term += -extension * Solid.spring_constant * dirn

        # Damping force to avoid constant oscillation
        force_term += -self.velocity * Solid.damping_constant
        
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
        Format for compressing each Solid instance into CSV.
        '''
        # Individual child instance info
        rest = ['*', self.position[0], self.position[1], \
                self.last_position[0],self.last_position[1],
                self.velocity[0], self.velocity[1],
                self.acceleration[0], self.acceleration[1] ]
        # Add connected links after id and before rest
        return [self.id, '*'] + self.connected_list + rest

    def read_csv_list(self, system_state_list, idx_shift):
        '''
        Format for parsing the compressed Solid instances from CSV.
        '''
        # Pass over starting '*'
        idx_shift += 2

        # Iterate through connected list
        self.connected_list = []
        while True:
            if system_state_list[idx_shift] == '*':
                break
            self.connected_list += [int(system_state_list[idx_shift])]
            idx_shift += 1

        # Read the rest idx_shift = 0 corresponds to '*'
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

    def instance_plot(self, ax, com=None, scale=None):
        ''' 
        Plots individual Solid particle onto existing axis, and plot its links
        '''
        # Get plot position of self in frame with COM
        plot_position = self.position
        size = 12**2 # 15**2
        default_line_width = 2
        if (com is not None) and (scale is not None):
            plot_position = self.orient_to_com(com, scale)
            size = np.max([size*Particle.walls_x_lim/scale,1])
        
        # Plot all links
        for other_id in self.connected_list:
            other = Solid.get_instance_by_id(other_id)
            # Check if link is stressed, colour differently
            colour = 'k'
            length = np.sqrt(self.dist(other))
            if length > Solid.spring_length * 2:
                colour = 'w'
            if length > Solid.spring_length * 1.1:
                colour = 'y'
            elif length < Solid.spring_length * 0.9:
                colour = 'r'
            # Get plot position (changes if COM scaled)
            other_plot_position = other.position
            linewidth = default_line_width
            markersize = 1
            if (com is not None) and (scale is not None):
                other_plot_position = other.orient_to_com(com, scale)
                linewidth = np.max([linewidth*Particle.walls_x_lim/scale,1])
                markersize = np.max([markersize*Particle.walls_x_lim/scale,1])
            # Plot link
            xvals = [plot_position[0], other_plot_position[0]]
            yvals = [plot_position[1], other_plot_position[1]]
            ax.plot(xvals, yvals,linestyle=':', marker='o', linewidth=linewidth, markersize=markersize, color=colour, zorder=self.id)
        
        # Plot main point on top (use higher zorder)
        ax.scatter(plot_position[0],plot_position[1],s=size,c='b', zorder=1000+self.id)


   