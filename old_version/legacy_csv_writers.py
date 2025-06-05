import numpy as np

'''
Legacy system encodes state log history into CSV with very specific encoding system.
This wasn't flexible or scaleable enough, so we switched to NDJSON
'''

class Particle:
    # -------------------------------------------------------------------------
    # CSV utilities

    @staticmethod
    def write_state_to_csv():
        '''
        Takes Particle system state at the current time, and compresses into CSV.
        Iterates through each class, and within that each class instance.
        Calls each class's own method to write its own section.
        '''
        #--------------------------------
        if not Particle.csv_path.endswith('.csv'):
            Particle.csv_path += '.csv'

        # Compose CSV row entry
        system_state_list = [Particle.current_step, Particle.current_time]

        # Iterate through all current child classes
        for classname in Particle.pop_counts_dict.keys():

            # Access the class directly from the all_instances dictionary
            class_instances = Particle.all.get(classname, {})

            # Initialise class specific list
            class_list = [classname, Particle.pop_counts_dict[classname]]

            # Iterate through all instances of the child class
            for child in class_instances.values():
                if child.alive == 1:
                    # Add instance info to list using its write_csv_list function
                    class_list += child.write_csv_list()

            # Add child class info to main list
            class_list += ['|']
            system_state_list += class_list

        # End CSV row with 'END'
        system_state_list += ['END']

        # ------------------------------------
        # Writing entry to file

        if not os.path.exists(Particle.csv_path):
            with open(Particle.csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                header_row = ['Timestep', 'Time', 'ClassName', 'ClassPop', 'InstanceID', 'Attributes', '...','|','ClassName','...','|','END']
                writer.writerow(header_row)
                writer.writerow(system_state_list)
        else:
            with open(Particle.csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(system_state_list)

    @staticmethod
    def load_state_from_csv(timestep):
        '''
        Reads from a CSV containing the compressed Particle system state at a specific time.
        Iterates through each class, and within that each class instance.
        Parses to decompress the format outlined in write_state_to_csv.
        '''
        # ------------------------------------
        # Read row from CSV

        with open(Particle.csv_path, mode='r', newline='') as file:
            # Loop through the CSV rows until reaching the desired row
            # (This must be done since CSV doesn't have indexed data structure)
            reader = csv.reader(file)
            target_row_index = timestep+1 
            for i, row in enumerate(reader):
                if i == target_row_index:
                    system_state_list = row.copy()
                    break
        
        # ------------------------------------
        # Parse row into a full Particle system state

        # Parse timestep info, shift index
        Particle.current_step, Particle.current_time = system_state_list[0], system_state_list[1]
        idx_shift = 2

        # Loop through blocks for each child class
        while True:
            # Check if reached the end of row
            if system_state_list[idx_shift] == 'END':
                break
            
            # Parse class and number of instances, shift index
            classname = system_state_list[idx_shift]
            class_pop = int(system_state_list[idx_shift+1])
            idx_shift += 2

            # Diagnostic print
            #print(f"Timestep {timestep}, class_pop {class_pop}, classname {classname}")

            # TODO: Note this structure is faulty - can't run if we start with 0 instances of a class
            if class_pop == 0:
                # Remove all instances from current system
                Particle.pop_counts_dict[classname] = 0
                Particle.max_ids_dict[classname] = -1
                Particle.all[classname] = {}
            else:
                '''
                # Search for existing instances
                existing_id = None
                for key, value in Particle.all[classname].items():
                    existing_id = key
                if existing_id is not None:
                    # We have a valid instance, clone it into prototype for future use
                    prototype = copy.copy(Particle.all[classname][existing_id])
                    Particle.prototypes[classname] = prototype
                '''
                # Cull everything and start again with prototype, rebuild with CSV info
                Particle.pop_counts_dict[classname] = 0
                Particle.max_ids_dict[classname] = -1
                Particle.all[classname] = {}
                prototype = Particle.prototypes[classname]

                # Create class_pop many clones by looping through CSV row
                for i in range(class_pop):
                    id = int(system_state_list[idx_shift])
                    # Clone our prototype with it's child class's create_instance method
                    child = prototype.create_instance(id=id)

                    # Assign attributes by reading the system_state_list for that class
                    # This calls to child class's method to read each instance
                    idx_shift = child.read_csv_list(system_state_list, idx_shift)
                    
                # Check for pipe | at the end, then move past it
                if system_state_list[idx_shift] != '|':
                    raise IndexError(f"Something wrong with parsing, ~ column {idx_shift}.")
            
            # Move on to next class by shifting over pipe character '|'
            idx_shift += 1


class Star:
    # -------------------------------------------------------------------------
    # CSV utilities - Legacy functions but working

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
    
class Prey:
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

class Predator:
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

class Solid:
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
    
class Pool:
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
    
class Human:
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