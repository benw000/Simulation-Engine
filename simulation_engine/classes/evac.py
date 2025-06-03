import numpy as np
import matplotlib.pyplot as plt
from .parents import Particle, Wall, Target
from simulation_engine.utils.manager import Manager



def setup(args):
    if args.deltat is None:
        args.deltat=Human.DEFAULT_TIMESTEP
    show_graph = False
    # Create manager instance
    manager = Manager(args = args, 
                      show_graph = show_graph,
                      draw_backdrop_plt_func = draw_backdrop_plt,
                      draw_graph_plt_func = draw_graph_plt)
    
    # Add Prey, Predator child classes to registry
    manager.class_objects_registry["Human"] = Human
    manager.class_objects_registry["Wall"] = Wall 
    manager.class_objects_registry["Target"] = Target
    manager.state["Environment"]["Wall"] = []
    manager.state["Environment"]["Target"] = []

    # Split by mode
    if args.mode == 'run':
        return setup_run(args, manager)
    elif args.mode == 'load':
        return manager
    
def setup_run(args, manager):
    # Set Particle geometry attributes
    Particle.env_x_lim = 12
    Particle.env_y_lim = 10
    Particle.track_com = False
    Particle.torus = False
    
    # Initialise walls
    x = Particle.env_x_lim
    y = Particle.env_y_lim
    wall_points = [[[0,0],[0,y]], # left wall
                    [[0,0],[x-0,0]], # bottom wall
                    [[0,y],[x-0,y]], # top wall
                    [[x-2,3.5],[x-2,6.5]], # big desk
                    [[x-0,0],[x,2]], # right wall bottom 
                    [[x,3],[x,7]], # right wall middle
                    [[x,8],[x-0,y]], # right wall top
                    #[[3,2],[3,4]],
                    #[[3,6],[3,8]],
                    [[4,2],[4,4]],
                    [[4,6],[4,8]],
                    #[[7,2],[7,4]],
                    #[[7,6],[7,8]],
                    [[7,2],[7,4]],
                    [[7,6],[7,8]]]
    for pair in wall_points:
        Wall(np.array(pair[0]), np.array(pair[1]))

    # Targets for each door
    Target(np.array([x+1,2.5]))
    Target(np.array([x+1,7.5]))

    # Spawn in evacuees
    for i in range(args.nums[0]):
        Human()

    return manager

def draw_backdrop_plt(ax):
    # Get an ax from manager, and plot things on it related to this mode
    # Overrides Manager.default_draw_backdrop_plt

    # Set padded limits
    ax.set_xlim(-1, Particle.env_x_lim+1)
    ax.set_ylim(-1, Particle.env_y_lim+1)

    # White background
    ax.set_facecolor('w')
    return ax

def draw_graph_plt(ax2):
    # TODO: Fix up
    # Draw a graph
    ax2.clear()
    max_time = int(self.num_timesteps)*self.delta_t
    ax2.set_xlim(0, max_time)  # Set x-axis limits
    ax2.set_ylim(0, Particle.num_evacuees) 
    xticks = [i for i in range(int(max_time)) if i % 5 == 0] + [max_time]  # Positions where you want the labels
    ax2.set_xticks(xticks)  # Set ticks at every value in the range
    ax2.set_xlabel("Time (s)")
    ax2.set_title(f"Number evacuated over time")
    ax2.set_aspect(aspect=Particle.env_y_lim/Particle.env_x_lim)
    t_vals = []
    y_vals = []
    for key, item in Particle.kill_record.items():
        if key <= timestep:
            t_vals += [int(key)*Particle.delta_t]
            y_vals += [item]
    ax2.plot(t_vals, y_vals, c='b')
    ax2.scatter(int(timestep)*Particle.delta_t,Particle.kill_record[timestep], marker='x', c='r')



class Human(Particle):
    '''
    Human particle for crowd simulation.
    '''
    # -------------------------------------------------------------------------
    # Attributes

    DEFAULT_TIMESTEP = 0.05
    personal_space = 0.5 # metres - 2 rulers between centres
    personal_space_repulsion = 300 # Newtons
    wall_dist_thresh = 0.2
    wall_repulsion = 2000
    wall_deflection = 3000
    # Constant force from each human to their target
    target_attraction = 200
    random_force = 100
    
    # Initialisation
    def __init__(self, position: np.ndarray = None, velocity: np.ndarray = None, id=None) -> None:
        '''
        Initialises a Human, inheriting from the Particle class.
        '''
        super().__init__(position, velocity, id)

        # Human specific attributes
        self.mass = 50
        self.max_speed = 1.5

        # Imprint on nearest target
        self.my_target = Target.find_closest_target(self)

    # -------------------------------------------------------------------------
    # Distance utilities

    def wall_deflection_dirn(self, wall_dist, wall_dirn):
        target_dist, target_dirn = self.dist(self.my_target, return_both=True)
        angle = np.arccos(np.dot(-wall_dirn,target_dirn)/(wall_dist*target_dist))
        tolerance = 1e-6
        if angle > (-np.pi / 2 + tolerance) and angle < 0:
            force_dirn = np.matmul(np.array([[0,-1],[1,0]]),wall_dirn)/wall_dist
            return force_dirn * self.wall_deflection # * np.cos(0.25*angle)
        elif angle>= 0 and angle<np.pi/2:
            force_dirn = np.matmul(np.array([[0,1],[-1,0]]),wall_dirn)/wall_dist
            return force_dirn * self.wall_deflection # * np.cos(0.25*angle)
        else:
            return np.zeros(2)


    # -------------------------------------------------------------------------
    # Main force model 

    def update_acceleration(self):
        '''
        Calculates main acceleration term from force-based model of environment.
        '''
        # Reconsider target every 20 timesteps
        if self.manager.current_step % 5 == 0:
            self.my_target = Target.find_closest_target(self)

        # Instantiate force term
        force_term = np.zeros(2)

        # Go through targets and check distance to escape threshold
        # If escape possible, unalive self. Otherwise sum my_target's force contribution
        for target in self.manager.state["Environment"]["Target"]:
            dist, dirn = self.dist(target, return_both=True)
            if dist < target.capture_thresh:
                self.unalive()
                return 1
            elif target is self.my_target:
                force_term += self.target_attraction * dirn #/dist

        # Human repulsion force - currently scales with 1/d
        for human in Human.iterate_class_instances():
            if human == self:
                continue
            elif self.dist(human) < self.personal_space:
                force_term += - self.unit_dirn(human)*(self.personal_space_repulsion/(np.sqrt(self.dist(human))))
                pass

        # Repulsion from walls - scales with 1/d^2
        for wall in self.manager.state["Environment"]["Wall"]:
            dist, dirn = wall.dist_to_wall(self)
            if dist < self.wall_dist_thresh:
                force_term += dirn * (self.wall_repulsion/(dist))
                # Make Humans smart - repel sideways if vector to target is directly blocked by wall
                if dist < self.wall_dist_thresh: #*0.5:
                    force_term += self.wall_deflection_dirn(dist, dirn)

        # Random force - stochastic noise
        # Generate between [0,1], map to [0,2] then shift to [-1,1]
        force_term += ((np.random.rand(2)*2)-1)*self.random_force

        # Update acceleration = Force / mass
        self.acceleration = force_term / self.mass

        return 0
    
    # -------------------------------------------------------------------------
    # CSV utilities

    # def write_csv_list(self):
    #     '''
    #     Format for compressing each Human instance into CSV.
    #     '''
    #     # Individual child instance info
    #     return [self.id, \
    #             self.position[0], self.position[1], \
    #             self.last_position[0],self.last_position[1],
    #             self.velocity[0], self.velocity[1],
    #             self.acceleration[0], self.acceleration[1] ]

    # def read_csv_list(self, system_state_list, idx_shift):
    #     '''
    #     Format for parsing the compressed Human instances from CSV.
    #     '''
    #     self.position = np.array([float(system_state_list[idx_shift+1]), \
    #                                 float(system_state_list[idx_shift+2])])
    #     self.last_position = np.array([float(system_state_list[idx_shift+3]), \
    #                                 float(system_state_list[idx_shift+4])])
    #     self.velocity = np.array([float(system_state_list[idx_shift+5]), \
    #                                 float(system_state_list[idx_shift+6])])
    #     self.acceleration = np.array([float(system_state_list[idx_shift+7]), \
    #                                 float(system_state_list[idx_shift+8])])
    #     # Update idx shift to next id and return
    #     return idx_shift+9
    
    # NDJSON
    def to_dict(self):
        new_dict = super().to_dict()
        new_dict["max_speed"] = self.max_speed
        return new_dict
    
    @classmethod
    def from_dict(cls, dict):
        instance = cls(position=np.array(dict["position"]),
                   velocity=np.array(dict["velocity"]),
                   unlinked=True)
        instance.acceleration = np.array(dict["acceleration"])
        instance.last_position = np.array(dict["last_position"])
        instance.mass = dict["mass"]
        instance.alive = dict["alive"]

        return instance

    # -------------------------------------------------------------------------
    
    # ---- MATPLOTLIB ----
    def draw_plt(self, ax:plt.Axes, com=None, scale=None):
        '''
        Updates the stored self.plt_artist PathCollection with new position
        '''
        # Remove artist from plot if dead
        if not self.alive:
            return self.remove_from_plot_plt()
        
        # Get plot position in frame
        plot_position = self.orient_to_com(com, scale)

        # Update artist with PathCollection.set_offsets setter method
        if self.plt_artists is None:
            # Initialise PathCollection artist as scatter plot point
            self.plt_artists = []
            self.plt_artists.append(ax.scatter(plot_position[0],plot_position[1], s=12**2,c='b'))
        else:
            # Update with offset
            self.plt_artists[0].set_offsets(plot_position)
        
        return self.plt_artists

