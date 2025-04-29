import copy
import numpy as np
from .parents import Particle, Environment, Wall, Target

class Pool(Particle):
    '''
    Solid particle for the lattice of springs simulation
    '''
    # -------------------------------------------------------------------------
    # Attributes

    # Constants
    Particle.delta_t=0.01
    damping_constant = 0.1
    d = 0.051 # 5.1 cm diameter of pool ball
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
                id=None) -> None:
        '''
        Initialises a solid object, inheriting from the Particle class.
        '''
        super().__init__(position, velocity,id)

        self.mass = 0.14
        self.colour = colour

        # Ensure prototype for child class exists, callable by its name as a string only
        prototype = copy.copy(self)
        Particle.prototypes[self.__class__.__name__] = prototype

    def create_instance(self,id):
        ''' Used to create instance of the same class as self, without referencing class. '''
        return Pool(id=id)
    
    def unalive(self):
        ''' 
        Sets the class instance with this id to be not alive, decrements the class count.
        '''
        self.alive=0
        Particle.pop_counts_dict[self.__class__.__name__] -= 1
        Particle.kill_count += 1
        Particle.kill_record[Particle.current_step] = Particle.kill_count
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
        Environment.background_type = "pool"
        Particle.walls_x_lim = 2
        Particle.walls_y_lim = 1

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
        up = Pool.d * np.array([-(np.sqrt(3)/2), 0.5])
        down = Pool.d * np.array([-(np.sqrt(3)/2), -0.5])

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
        if Environment.targets is not []:
            for target in Environment.targets:
                dist, _ = target.dist_to_target(self)
                if dist < self.d:
                    self.unalive()
                    return 1

        # Repulsion from balls - in range scales with 1/d up to limit
        for ball in Pool.iterate_class_instances():
            if ball == self:
                continue
            dist = np.sqrt(self.dist(ball))
            if dist < self.d - 0.0005: #0.001 1 diameter between ball centres -> collision
                #print(f"Collision detected between {self.id} and {ball.id}, distance {dist} metres")
                repulsion = np.min( [self.k_ball/(dist), self.max_collision_force] )
                force_term += - self.unit_dirn(ball)*repulsion
                self.just_reflected = True
            else:
                continue

        # Reflection from walls - in range, walls act like stiff springs
        for wall in Environment.walls:
            dist, _ = wall.dist_to_wall(self)
            if dist < 0.5*self.d: # 1 radius to wall -> collision F = ke try 1
                compression = self.d - dist
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
    
    # -------------------------------------------------------------------------
    # Animation utilities

    def instance_plot(self, ax, com=None, scale=None):
        ''' 
        Plots individual Pool ball particle onto existing axis, and plots score
        '''
        size = 8**2
        ax.scatter(self.position[0], self.position[1], color=self.colour, s=size)

        # Plot scores if not already done
        if ax.texts:
            pass
        else:
            fontsize = 5
            if Pool.num_red_potted > 0:
                ax.text(x=0.2,y=-Pool.d, s=f"{Pool.num_red_potted}", fontsize=fontsize, color='r')
            if Pool.num_yellow_potted > 0:
                ax.text(x=0.4,y=-Pool.d, s=f"{Pool.num_yellow_potted}", fontsize=fontsize, color='y')
            if Pool.white_potted:
                ax.text(x=0.6,y=-Pool.d, s=f"{Pool.white_potted}", fontsize=fontsize, color='w')
            if Pool.black_potted:
                ax.text(x=0.8,y=-Pool.d, s=f"{Pool.black_potted}", fontsize=fontsize, color='k')






