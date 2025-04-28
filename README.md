# Simulation Engine

Oct 2024 \
Ben Winstanley

----

<table>
  <tr> 
    <td><img src="https://github.com/benw000/Simulation-Engine/blob/main/demo_videos/pool_demo.gif" alt="Pool tile" width = "300"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/benw000/Simulation-Engine/blob/main/demo_videos/evac_no_graph_demo.gif" alt="Evac tile" width="300"/></td>
    <td><img src="https://github.com/benw000/Simulation-Engine/blob/main/demo_videos/springs_demo.gif" alt="Springs tile" width="300"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/benw000/Crowd-Simulation/blob/main/demo_videos/birds_demo.gif" alt="Birds tile" width="300"/></td>
    <td><img src="https://github.com/benw000/Crowd-Simulation/blob/main/demo_videos/nbody_demo.gif" alt="Nbody tile" width="300"/></td>
  </tr>
</table>

### General physics simulation engine for force-based particle models, built from scratch in Python.

Follows object-oriented design: 
- The main 'Particle' class handles all numerical integration, writing/reading CSV logs, animating, compositing to MP4 etc..
- Each of its child classes (eg. 'Person', 'Star', 'Solid',...) specifies a set of forces describing a dynamical system, along with plotting appearance. 

This is an active project - I'm still working on new systems, as well as working to improve the modular design. My aim is to make it quick and easy for a user to specify, and then simulate a completely new dynamical system (with all the annoying details obscured on the backend).

---

## Examples:

### 8-ball pool breaking simulation

![pool-gif](https://github.com/benw000/Simulation-Engine/blob/main/demo_videos/pool_demo.gif)

Run this : ```python general_simulation.py --type pool --steps 500 ```

Pool balls are initialised in the normal setup, the cue ball starting off firing into the triangle with a slight random vertical deviance. Balls repel off eachother, simulating elastic collision, and reflect off of the cushion walls, being removed if they hit the target of any pocket.

**Forces**
- Repulsion force between contacting balls - very strong but active within a small range, scaling with 1/d.
- Normal force from wall - this models each cushion as a spring, with any compression from incoming balls resulting in an outwards normal force on the ball, following Hooke's law.


### Classroom Evacuation Model


![evac-gif](https://github.com/benw000/Crowd-Simulation/blob/main/demo_videos/evac_demo.gif)

Run this :  ``` python general_simulation.py --type evac --num 40 --steps 200 ```

People are initialised at random points in the classroom, and make their way to the nearest exit, periodically re-evaluating which exit to use. The graph on the right shows the number of people evacuated over time, which can be used to score different classroom layouts for fire safety. Layouts are easily created by specifying pairs of vertices for each wall (seen in red), and specifying a number of target locations (green crosses).

**Forces**
- Constant attraction force to an individual's chosen target exit.
- Repulsion force between people - active within a personal space threshold, scales with $\frac{1}{Distance}$
- Repulsion force from walls - also active within a threshold, scales with $\frac{1}{Distance^2}$
- Deflection force from walls - force acting along length of wall towards an individual's target, prevents gridlock when a wall directly obscures the target.
- Stochastic force - a small amount of noise is applied.
- Note that additional bespoke forces would have to be specified in order to encode more intelligent, calculating behaviour.

---

### Spring System Model

![springs-gif](https://github.com/benw000/Simulation-Engine/blob/main/demo_videos/springs_demo.gif)

Run this :  ``` python general_simulation.py --type springs --num 50  --steps 30 ```

Point particles are initialised at random positions on the plane; if a neighbour is within a spring length away, a spring is formed. Particles with no connections are culled before step 0. We see larger molecules start to form as networks of connected particles reach an equillibrium. Setting a larger spring length allows more particles to connect to eachother, increasing the complexity of the structures formed.

**Forces** 
- Elastic force following Hooke's law: $F = -k \cdot (Spring \  Extension)$. This acts on both particles whenever the spring between them is in compression (red), or extension (yellow).
- Damping force - directly opposes particle motion, scaling linearly with velocity.
- Stochastic force - a small amount of random noise is applied to each particle.

---

### Predator-Prey Model

![birds-gif](https://github.com/benw000/Crowd-Simulation/blob/main/demo_videos/birds_demo.gif)

Run this :  ``` python general_simulation.py --type birds --num 50 --num2 5 --steps 200```

Prey (white) and predator (red) birds are initialised at random points. The predators act to hunt each prey, always pursuing the closest bird and killing it within a certain distance threshold. The prey avoid the predators, and flock together to increase chances of survival. This all takes place on a torus topological domain, where opposite edges are connected with periodic boundary conditions. The predators aren't particularly intelligent, since their motion is governed by simple blind attraction at each timestep.

**Forces**
- Constant attraction force on predators towards closest prey, and repulsion on prey away from all predators.
- Constant attraction force on prey towards the centre of mass of all prey birds - this encodes flocking behaviour.
- Repulsion force between birds - active within a personal space threshold, scales with $\frac{1}{Distance}$.
- Stochastic force - a fair amount of noise is applied to the prey, to simulate erratic movements to throw off predators. Less noise is applied to the predators, which are very direct.

--- 

### N-body Gravitational Dynamics


![nbody-gif](https://github.com/benw000/Crowd-Simulation/blob/main/demo_videos/nbody_demo.gif)

Run this : ``` python general_simulation.py --type nbody --num 30 --steps 100 ```

Bodies are initialised with random positions and velocities, and masses of different magnitudes, chosen from a log-uniform distribution scale. Each body feels a gravitational attraction towards every other body in the system. Larger bodies attract smaller ones, which accelerate towards them. To a first order level of approximation, these smaller bodies then engage in elliptic orbits around the larger body, or are deflected, shooting off on a parabolic trajectory. As more bodies shoot off, their density in our viewing window decreases.

**Forces**:
- Gravitational attraction force - each body is attracted to every other body in the system, following Newton's law of universal gravitation: $F = G \frac{Mass_1 Mass_2}{Distance^2}$.

---






### Files:
- classes.py : contains main Particle class, with Prey, Predator, Star, Human child classes, and Environment class with Wall child class.
- simulation_classes : module folder containing seperated files with each of these classes
- general_simulation.py : calls objects from classes.py, wraps up in simple time-stepping script in main(). Run with commands specified above.
- old_version : folder containing older version of project
- demo_videos : folder containing demo videos and gifs
- VersionControl.md : markdown document with Git commands and explanation
- .gitignore : file telling Git to ignore output CSV and MP4 folders
- publish_all.sh : bash script to automatically push to main (Use carefully)

---

### Starting Aims:

- Create a crowd simulation script, which will take a number of pedestrians and model their movement with a force-based model
- This should take initial positions, a custom 2D environment, and update each pedestrians position and velocity at each timestep
- This should be done following Git and Python best practices, with object-oriented programming. Use git terminal commands, and commit from develop branch to main.
- This should store information in an updated CSV file for each run.
- At the end of the computation, the script should read the CSV by line and create an mp4 video.

### Extensions:
- Cloth simulation using perturbed SHM oscillator model for each particle in lattice.
- Evac sim fire instances
- Evac sim where we fuse force model with a neural network to work out velocity. Reinforcement learning with PyTorch to train:
    1. Selfish NN behaviour to minimise personal time until evacuated
    2. Collective NN behaviour to minimise total time until all evacuated
- If this works, do the same with birds in torus world, passing torus shortest dist and dirn. We then could have a GAN between predator and prey, both competing.
- Port all of this to an interactive HTML page which has a Python backend.

---
