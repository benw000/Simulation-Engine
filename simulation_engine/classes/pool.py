import numpy as np
from .parents import Particle, Wall, Target
from simulation_engine.utils.manager import Manager

def setup(args):
    if args.deltat is None:
        args.deltat=Pool.DEFAULT_TIMESTEP
    # Create manager instance
    manager = Manager(args = args, 
                      show_graph = False,
                      draw_backdrop_plt_func = draw_backdrop_plt)
    
    # Add Prey, Predator child classes to registry
    manager.class_objects_registry["Pool"] = Pool
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
    # ---- VALIDATE ARGS ----

    # Set Particle geometry attributes
    Particle.track_com = False
    Particle.torus = False

    # Initialise particles and environment
    Pool.pool_setup()

    return manager

def draw_backdrop_plt(ax):
    # Get an ax from manager, and plot things on it related to this mode
    # Overrides Manager.default_draw_backdrop_plt

    # Set padded limits
    ax.set_xlim(-1, Particle.env_x_lim+1)
    ax.set_ylim(-1, Particle.env_y_lim+1)

    # Black background
    ax.set_facecolor('lightgreen')

    # TODO: Check these limits
    ax.set_xlim([-2*0.05, 2+2*0.05])
    ax.set_ylim([-2*0.05, 1+2*0.05])

    # Plot head string of pool table
    ax.plot([1.56,1.56],[0,1], c='w', alpha=0.5, linestyle=':')
    # Plot foot spot
    ax.scatter(0.64,0.5, c='w', s=10)

    return ax

class Pool(Particle):
    '''
    Solid particle for the lattice of springs simulation
    '''
    # -------------------------------------------------------------------------
    # Attributes

    plot_text = False

    # Constants
    DEFAULT_TIMESTEP = 0.01
    damping_constant = 0.1
    diameter = 0.051 # 5.1 cm diameter of pool ball
    k_ball = 0.3 #0.5
    k_wall = 1000
    max_collision_force = 5 # limit collision force by 50N upper limit

    # Counts
    num_red_potted = 0
    num_yellow_potted = 0
    white_potted = 0
    black_potted = 0

    # Initialisation
    def __init__(self, colour: str = 'r', position: np.ndarray = None, velocity: np.ndarray = None,
                unlinked=False) -> None:
        '''
        Initialises a solid object, inheriting from the Particle class.
        '''
        super().__init__(position, velocity, unlinked)

        self.mass = 0.14
        self.colour = colour
    
    def unalive(self):
        ''' 
        Sets the class instance with this id to be not alive, decrements the class count.
        '''
        super().unalive()
        # Update colour records
        if self.colour == 'r':
            Pool.num_red_potted += 1
        elif self.colour == 'y':
            Pool.num_yellow_potted += 1
        elif self.colour == 'w':
            Pool.white_potted = 1
        elif self.colour == 'k':
            Pool.black_potted = 1

    def __str__(self):
        return f"Pool ball {self.id}, colour {self.colour} at position {self.position} with velocity {self.velocity}"

    # -------------------------------------------------------------------------
    # Pool table setup

    def pool_setup():
        '''
        Sets up standard game of 8-ball 'pub' pool
        '''
        Particle.env_x_lim = 2
        Particle.env_y_lim = 1

        # Pocket widths
        corner_width, middle_width = 0.089, 0.102
        x = corner_width * (np.sqrt(2)/2)
        y = middle_width / 2

        # Pocket setup
        Target(np.array([0,0]), capture_radius=x) # Bottom foot corner
        Target(np.array([0,1]), capture_radius=x) # Top foot corner
        Target(np.array([2,0]), capture_radius=x) # Bottom head corner
        Target(np.array([2,1]), capture_radius=x) # Top head corner
        Target(np.array([1,1+y]), capture_radius=np.sqrt(2)*y) # Front middle
        Target(np.array([1,0-y]), capture_radius=np.sqrt(2)*y) # Bottom middle

        # Cushions
        
        # -- Note order of a,b matters: 'inside' normal is anticlockwise to a->b
        Wall.DEFAULT_EDGE_COLOUR = 'brown'
        Wall.DEFAULT_LINE_COLOUR = 'brown'
        Wall(np.array([0,1-x]), np.array([0,x])) # Foot cushion
        Wall(np.array([2,x]), np.array([2,1-x])) # Head cushion
        Wall(np.array([x,0]), np.array([1-y,0])) # Bottom left cushion
        Wall(np.array([1+y,0]), np.array([2-x,0])) # Bottom right cushion
        Wall(np.array([1-y,1]), np.array([x,1])) # Top left cushion
        Wall(np.array([2-x,1]), np.array([1+y,1])) # Top right cushion

        # Balls
        # -- White: behind head string, small amount of random vertical velocity
        white_speed = 4
        noise = (np.random.rand(1)*2 - 1)[0]*0.05
        Pool(colour='w', position=(np.array([1.7,0.5])), velocity=np.array([-white_speed,noise]))

        # -- Triangle setup - moving leftwards, top down
        apex = np.array([0.64,0.5])
        zero = np.array([0,0])
        up = Pool.diameter * np.array([-(np.sqrt(3)/2), 0.5])
        down = Pool.diameter * np.array([-(np.sqrt(3)/2), -0.5])

        # --- Row 0
        Pool(colour='r', position=apex, velocity=zero)
        # --- Row 1
        Pool(colour='y',position=apex+up, velocity=zero)
        Pool(colour='r',position=apex+down, velocity=zero)
        # --- Row 2
        Pool(colour='r',position=apex+2*up, velocity=zero)
        Pool(colour='k',position=apex+up+down, velocity=zero)
        Pool(colour='y',position=apex+2*down, velocity=zero)
        # --- Row 3
        Pool(colour='y',position=apex+3*up, velocity=zero)
        Pool(colour='r',position=apex+2*up+down, velocity=zero)
        Pool(colour='y',position=apex+up+2*down, velocity=zero)
        Pool(colour='r',position=apex+3*down, velocity=zero)
        # --- Row 4
        Pool(colour='r',position=apex+4*up, velocity=zero)
        Pool(colour='y',position=apex+3*up+down, velocity=zero)
        Pool(colour='r',position=apex+2*up+2*down, velocity=zero)
        Pool(colour='y',position=apex+up+3*down, velocity=zero)
        Pool(colour='r',position=apex+4*down, velocity=zero)


    # -------------------------------------------------------------------------
    # Main force model
    
    def update_acceleration(self):
        '''
        Calculates main acceleration term from force-based model of environment.
        '''
        # Instantiate force term
        force_term = np.zeros(2)

        # Go through targets and check distance to escape threshold
        # If escape possible, unalive self, and cease update function

        for target in self.manager.state["Environment"]["Target"]:
            dist = self.dist(target)
            if dist < self.diameter:
                self.unalive()
                return 1

        # Repulsion from balls - in range scales with 1/d up to limit
        for ball in Pool.iterate_class_instances():
            if ball == self:
                continue
            dist = np.sqrt(self.dist(ball))
            if dist < self.diameter - 0.0005: #0.001 1 diameter between ball centres -> collision
                #print(f"Collision detected between {self.id} and {ball.id}, distance {dist} metres")
                repulsion = np.min( [self.k_ball/(dist), self.max_collision_force] )
                force_term += - self.unit_dirn(ball)*repulsion
                self.just_reflected = True
            else:
                continue

        # Reflection from walls - in range, walls act like stiff springs
        for wall in self.manager.state["Environment"]["Wall"]:
            dist, _ = wall.dist_to_wall(self)
            if dist < 0.5*self.diameter: # 1 radius to wall -> collision F = ke try 1
                compression = self.diameter - dist
                force_term += wall.perp_vec * self.k_wall * compression*0.8
                self.just_reflected = True

        # Damping force to simulate friction
        force_term += -self.velocity * Pool.damping_constant

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
        return [self.id, self.colour, \
                self.position[0], self.position[1], \
                self.last_position[0],self.last_position[1],
                self.velocity[0], self.velocity[1],
                self.acceleration[0], self.acceleration[1] ]

    def read_csv_list(self, system_state_list, idx_shift):
        '''
        Format for parsing the compressed Star instances from CSV.
        '''
        self.colour = str(system_state_list[idx_shift+1])
        self.position = np.array([float(system_state_list[idx_shift+2]), \
                                    float(system_state_list[idx_shift+3])])
        self.last_position = np.array([float(system_state_list[idx_shift+4]), \
                                    float(system_state_list[idx_shift+5])])
        self.velocity = np.array([float(system_state_list[idx_shift+6]), \
                                    float(system_state_list[idx_shift+7])])
        self.acceleration = np.array([float(system_state_list[idx_shift+8]), \
                                    float(system_state_list[idx_shift+9])])
        # Update idx shift to next id and return
        return idx_shift+10
    
    # NDJSON
    def to_dict(self):
        new_dict = super().to_dict()
        new_dict["colour"] = self.colour
        return new_dict
    
    @classmethod
    def from_dict(cls, dict):
        instance = cls(position=np.array(dict["position"]),
                   velocity=np.array(dict["velocity"]),
                   unlinked=True)
        instance.acceleration = np.array(dict["acceleration"])
        instance.last_position = np.array(dict["last_position"])
        instance.mass = dict["mass"]
        instance.colour = dict["colour"]
        instance.alive = dict["alive"]
        return instance
    
    # -------------------------------------------------------------------------
    # Animation utilities

    def draw_plt(self, ax, com=None, scale=None):
        size = 8**2

        # Remove artist from plot if dead
        if not self.alive:
            return self.remove_from_plot_plt()
        
        # Get plot position in frame
        plot_position = self.orient_to_com(com, scale)

        # Update artist with PathCollection.set_offsets setter method
        if self.plt_artists is None:
            self.plt_artists = []
            fontsize = 5
            if self.plot_text:
                self.plt_artists.append(ax.text(x=0.2,y=-Pool.diameter, s=f"{Pool.num_red_potted}", 
                                                fontsize=fontsize, color='r'))
                self.plt_artists.append(ax.text(x=0.4,y=-Pool.diameter, s=f"{Pool.num_yellow_potted}", 
                                                fontsize=fontsize, color='y'))
                self.plt_artists.append(ax.text(x=0.6,y=-Pool.diameter, s=f"{Pool.white_potted}", 
                                                fontsize=fontsize, color='w'))
                self.plt_artists.append(ax.text(x=0.8,y=-Pool.diameter, s=f"{Pool.black_potted}", 
                                                fontsize=fontsize, color='k'))
            self.plt_artists.append(ax.scatter(plot_position[0], plot_position[1], color=self.colour, s=size))
        else:
            if self.plot_text:
                self.plt_artists[0].set_text(f"{Pool.num_red_potted}")
                self.plt_artists[1].set_text(f"{Pool.num_yellow_potted}")
                self.plt_artists[2].set_text(f"{Pool.white_potted}")
                self.plt_artists[3].set_text(f"{Pool.black_potted}")

            self.plt_artists[-1].set_offsets(plot_position)
                
        return self.plt_artists

'''
TODO:
Update init, unalive
Update update_acceleration to hook up to Manager
New Pool.to_dict(), from_dict() methods
Update instance_plot to draw_plt
Remove one of the x_lim setters
See if the Environment stuff is working or needs rehaul
Does environment stuff work with saving to video, logs, etc
Parts of it may need to be re-called at the start of loading from log!

'''