import numpy as np

from simulation_engine.utils.errors import SimulationEngineInputError
from simulation_engine.classes import Particle
from simulation_engine.utils.manager import Manager

# -------------------------------------------------------------------------
# Setup

def setup(args):
    """
    Called by main entrypoint script as entry into this simulation 'type' module.
    Divides between different setup functions for each different 'mode'

    Args:
        args (argparse.Namespace): argparse namespace of user supplied arguments
    """
    # Default timestep
    if args.deltat is None:
        args.deltat=Solid.DEFAULT_TIMESTEP

    # Create manager instance
    manager = Manager(args = args, 
                      show_graph = False,
                      draw_backdrop_plt_func = draw_backdrop_plt)
    
    # Add Prey, Predator child classes to registry
    manager.class_objects_registry["Solid"] = Solid

    # Split by mode
    if args.mode == 'run':
        return setup_run(args, manager)
    elif args.mode == 'load':
        return manager

def setup_run(args, manager):
    # Setup args
    if not len(args.nums) == 1:
        raise SimulationEngineInputError("(-n, --nums) Please supply 1 argument only for population when using springs simulation type")
    
    # Set Particle geometry attributes
    Particle.env_x_lim = 100
    Particle.env_y_lim = 100
    Particle.track_com = True
    Particle.torus = False

    # Initialise particles - could hide this in Particle but nice to be explicit
    for i in range(args.nums[0]):
        Solid()

    if not manager.state["Particle"].get("Solid", None):
        print("No spring particles survived after initialisation. Please add more or change spring length!")
        exit()
    return manager

def draw_backdrop_plt(ax):
    """
    Get an ax from manager, and plot things on it related to this mode
    Overrides Manager.default_draw_backdrop_plt

    Args:
        ax (plt.Axes): Main matplotlib frame
    """
    # Black background
    ax.set_facecolor('white')

class Solid(Particle):
    '''
    Solid particle for the lattice of springs simulation
    '''
    # -------------------------------------------------------------------------
    # Attributes

    # Default timestep
    DEFAULT_TIMESTEP = 0.05
    
    # Forces
    spring_length = 12 # use 12 or 10
    spring_constant = 3
    damping_constant = 2.5
    random_force = 0.01

    # Links
    links_count = 0
    links_dict = {}

    def __init__(self, position: np.ndarray = None, velocity: np.ndarray = None, unlinked=None) -> None:
        '''
        Initialises a solid object, inheriting from the Particle class.
        '''
        super().__init__(position, velocity, unlinked)
        self.mass = 1
        self.connected_list = []
        # Upon initialising new solid, link to points nearby
        if not unlinked:
            for other in Solid.iterate_class_instances():
                if other is self:
                    continue
                # Check distance is close enough to self
                if self.dist(other) < self.spring_length * 1.2:
                    # Form link between self, other, check its not already in links_dict
                    link = sorted([self.id, other.id]) # Sort to make link unique
                    if link in Solid.links_dict.values():
                        continue
                    # Update count, add to global dict
                    Solid.links_count += 1
                    Solid.links_dict[Solid.links_count] = link
                    # Update lists
                    self.connected_list += [other.id]
                    other.connected_list += [self.id]
        
    # -------------------------------------------------------------------------
    # Main force model
    
    def update_acceleration(self):
        '''
        Calculates main acceleration term from force-based model of environment.
        '''
        # On first timestep, if solid is not connected to others, kill it
        if self.manager.current_step == 0 and self.connected_list == []:
            self.unalive()
            
        # Instantiate force term
        force_term = np.zeros(2)

        # Spring force from links
        for other_id in self.connected_list:
            other = Solid.get_instance_by_id(other_id)
            # Get dist^2 and direction to other
            dist, dirn = self.dist(other, return_both=True)
            # Spring force term based on extension
            extension = Solid.spring_length - np.sqrt(dist)
            force_term += -extension * Solid.spring_constant * dirn

        # Damping force to avoid constant oscillation
        force_term += -self.velocity * self.damping_constant
        
        # Random force - stochastic noise
        # Generate between [0,1], map to [0,2] then shift to [-1,1]
        force_term += ((np.random.rand(2)*2)-1)*self.random_force

        # Update acceleration = Force / mass
        self.acceleration = force_term / self.mass

        return 0

    # -------------------------------------------------------------------------
    # Logging

    def to_dict(self):
        new_dict = super().to_dict()
        new_dict["connected_list"] = self.connected_list
        return new_dict
    
    @classmethod
    def from_dict(cls, dict):
        instance = cls(position=np.array(dict["position"]),
                   velocity=np.array(dict["velocity"]),
                   unlinked=True)
        instance.acceleration = np.array(dict["acceleration"])
        instance.last_position = np.array(dict["last_position"])
        instance.connected_list = dict["connected_list"]
        instance.alive = dict["alive"]
        return instance
    
    # -------------------------------------------------------------------------
    # Matplotlib

    def draw_plt(self, ax, com=None, scale=None):
        ''' 
        Plots individual Solid particle onto existing axis, and plot its links
        '''
        # Remove artist from plot if dead
        if not self.alive:
            return self.remove_from_plot_plt()
        
        # Get plot position of self in frame with COM
        plot_position = self.orient_to_com(com, scale)
        # Set default size
        size = 10**2 # 15**2
        if self.track_com:
            size = np.max([size*Particle.env_x_lim/scale,1])
        default_line_width = 2

        if self.plt_artists is None:
            self.plt_artists = []
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
                    linewidth = np.max([linewidth*Particle.env_x_lim/scale,1])
                    markersize = np.max([markersize*Particle.env_x_lim/scale,1])
                # Plot link
                xvals = [plot_position[0], other_plot_position[0]]
                yvals = [plot_position[1], other_plot_position[1]]
                self.plt_artists.append(ax.plot(xvals, yvals,linestyle=':',
                                                marker='o', linewidth=linewidth, 
                                                markersize=markersize, color=colour,
                                                zorder=self.id)[0])
            
            # Plot main point on top (use higher zorder)
            self.plt_artists.append(ax.scatter(plot_position[0],plot_position[1],s=size,c='b', zorder=1000+self.id))
        else:
            # Plot all links
            for idx, other_id in enumerate(self.connected_list):
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
                    linewidth = np.max([linewidth*Particle.env_x_lim/scale,1])
                    markersize = np.max([markersize*Particle.env_x_lim/scale,1])
                # Plot link
                xvals = [plot_position[0], other_plot_position[0]]
                yvals = [plot_position[1], other_plot_position[1]]
                self.plt_artists[idx].set_data(xvals, yvals)
                self.plt_artists[idx].set_color(colour)
                self.plt_artists[idx].set_markersize(markersize)
            
            # Plot main point on top (use higher zorder)
            self.plt_artists[-1].set_offsets(plot_position)
            self.plt_artists[-1].set_sizes([size])
        
        return self.plt_artists