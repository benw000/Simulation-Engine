import numpy as np
import copy

from .parents import Particle, Environment, Wall, Target

class Human(Particle):
    '''
    Human particle for crowd simulation.
    '''
    # -------------------------------------------------------------------------
    # Attributes

    personal_space = 1 # metres - 2 rulers between centres
    personal_space_repulsion = 300 # Newtons

    wall_dist_thresh = 0.5
    wall_repulsion = 200

    target_attraction = 1500

    random_force = 200
    
    # Initialisation
    def __init__(self, position: np.ndarray = None, velocity: np.ndarray = None, id=None) -> None:
        '''
        Initialises a Human, inheriting from the Particle class.
        '''
        super().__init__(position, velocity, id)

        # Human specific attributes
        self.mass = 60
        self.max_speed = 1.8

        # Imprint on nearest target
        if Environment.targets is not []:
            self.my_target = Target.find_closest_target(self)

        # Ensure prototype for child class exists, callable by its name as a string only
        prototype = copy.copy(self)
        Particle.prototypes[self.__class__.__name__] = prototype

    def create_instance(self,id):
        ''' Used to create instance of the same class as self, without referencing class. '''
        return Human(id=id)

    # -------------------------------------------------------------------------
    # Distance utilities

    def wall_deflection(self, wall_dist, wall_dirn):
        target_dist, target_dirn = self.my_target.dist_to_target(self)
        angle = np.arccos(np.dot(-wall_dirn,target_dirn)/(wall_dist*target_dist))
        tolerance = 1e-6
        if angle > (-np.pi / 2 + tolerance) and angle < 0:
            force_dirn = np.matmul(np.array([[0,-1],[1,0]]),wall_dirn)/wall_dist
            return force_dirn * self.wall_repulsion # * np.cos(0.25*angle)
        elif angle>= 0 and angle<np.pi/2:
            force_dirn = np.matmul(np.array([[0,1],[-1,0]]),wall_dirn)/wall_dist
            return force_dirn * self.wall_repulsion # * np.cos(0.25*angle)
        else:
            return np.zeros(2)


    # -------------------------------------------------------------------------
    # Main force model 

    def update_acceleration(self):
        '''
        Calculates main acceleration term from force-based model of environment.
        '''
        # Reconsider target every 20 timesteps
        if Particle.current_step % 10 == 0:
            self.my_target = Target.find_closest_target(self)

        # Instantiate force term
        force_term = np.zeros(2)

        # Go through targets and check distance to escape threshold
        # If escape possible, unalive self. Otherwise sum my_target's force contribution
        if Environment.targets is not []:
            for target in Environment.targets:
                dist, dirn = target.dist_to_target(self)
                if dist**2 < target.capture_thresh:
                    self.unalive()
                    return 1
                elif target is self.my_target:
                    force_term += self.target_attraction * (dirn/dist**2)

        # Human repulsion force - currently scales with 1/d^2
        for human in Human.iterate_class_instances():
            if human == self:
                continue
            elif self.dist(human) < self.personal_space:
                force_term += - self.unit_dirn(human)*(self.personal_space_repulsion/(np.sqrt(self.dist(human))))
                pass

        # Repulsion from walls - scales with 1/d^2
        for wall in Environment.walls:
            dist, dirn = wall.dist_to_wall(self)
            if dist < self.wall_dist_thresh:
                force_term += dirn * (self.wall_repulsion/(dist**2))
                # Make Humans smart - repel sideways if vector to target is directly blocked by wall
                if dist < 0.5 * self.wall_dist_thresh:
                    force_term += 3*self.wall_deflection(dist, dirn)

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
        Format for compressing each Human instance into CSV.
        '''
        # Individual child instance info
        return [self.id, \
                self.position[0], self.position[1], \
                self.last_position[0],self.last_position[1],
                self.velocity[0], self.velocity[1],
                self.acceleration[0], self.acceleration[1] ]

    def read_csv_list(self, system_state_list, idx_shift):
        '''
        Format for parsing the compressed Human instances from CSV.
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

    def instance_plot(self, ax, com=None, scale=None):
        ''' 
        Plots individual Prey particle onto existing axis. 
        '''
        # Get plot position in frame
        plot_position = self.position
        if (com is not None) and (scale is not None):
            plot_position = self.orient_to_com(com, scale)
        
        ax.scatter(plot_position[0],plot_position[1],s=12**2,c='b')

        


