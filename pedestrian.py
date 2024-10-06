# person.py

import numpy as np

# Define global counter variable
counter = 0

class Person:
    # Class attribute set of all people
    all = []
    
    # Person
    person_dist_thresh = 100**2
    person_force = 5
    max_speed = 5
    person_inertia = 1

    # Walls 
    walls_dist_thresh = 1*2
    walls_force = 10
    walls_x_lim = 100
    walls_y_lim = 100

    # Random
    random_force = 1

    # Class to describe each person in simulation
    def __init__(self, position: np.ndarray, velocity: np.ndarray) -> None:
        # Check for correct formats

        # Assign defaults
        self.position = position
        self.velocity = velocity
        self.force_term = np.zeros(2)

        # Add to set of all people
        global counter
        self.person_id = counter
        counter += 1
        all += [self]

    # Print statement for person
    def __str__(self) -> str:
        return f"Person at position {self.position} with velocity {self.velocity}"

    # Debug statement for person
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.position},{self.velocity})"

    # Calculate squared euclidean distance between people
    def dist(self,other) -> float:
        return (self.position-other.position)**2
    
    # Calculate distance direction between people
    def dirn(self,other) -> float:
        return (self.position-other.position)/np.sqrt(self.dist(other))
    
    # Update force term and store as attribute
    def update_force_term(self) -> None:
        force_term = np.zeros(2)

        # Personal force
        for person in all:
            if person == self:
                continue
            elif self.dist(person) < Person.person_dist_thresh:
                force_term += self.dirn(person)*(Person.person_force/(self.dist(person)))
            else:
                continue

        # Force from walls - ideally would find shortest dist etc
        person_x, person_y = self.position[0], self.position[1]
        if person_x < Person.walls_dist_thresh:
            force_term += Person.walls_force * np.array([1,0])
        elif person_x > (Person.walls_x_lim - Person.walls_dist_thresh):
            force_term += Person.walls_force * np.array([-1,0])
        if person_y < Person.walls_dist_thresh:
            force_term += Person.walls_force * np.array([0,1])
        elif person_y > (Person.walls_y_lim - Person.walls_dist_thresh):
            force_term += Person.walls_force * np.array([0,-1])

        # Random force
        force_term += np.random.rand(2)*Person.random_force

        self.force_term = force_term

    def update_velocity(self):
        # Update velocity via force term
        self.velocity += self.person_inertia * self.force_term
        speed = np.sqrt(np.sum(self.velocity)**2)
        # Normalise speed to max in whichever direction it points
        if speed > Person.max_speed:
            self.velocity *= Person.max_speed/speed



            


    


    


    
    