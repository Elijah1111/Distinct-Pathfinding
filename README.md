

# Distinct-Pathfinding
A machine learning model that pathfinds on height maps

# Files
perlin.py: Noise generation for creating the height mappings
omplExample.py: Testing bed for running ompl implementations
rrt.py: Where most of the implementation of RRT exists
rrtstar.py: Where most of the implementation of RRT* exists
train.py: The main implementation of code including the reinforcement model and calculations (Change SIZE variable to any size divisible by 2 here and in perlin.py to change the size of the height mapping)

# Libraries
* [FastNoiseSIMD](https://github.com/Auburn/FastNoiseSIMD)

  A parallel proscessing library for generating various noise types quickly

* [pyfastnoisesimd](https://github.com/robbmcleod/pyfastnoisesimd)

  A python wrapper for FastNoise allowing usage in python project

* [ompl-thin](https://github.com/HiroIshida/ompl-thin-python)

  A python wrapper for OMPL to implement RRT

* [pathfinding](https://github.com/brean/python-pathfinding)

  An implementation of classic planning algorithms for comparison

# Installation 

 Install [OMPL](https://ompl.kavrakilab.org/)
 * Just install the bindings, we are using a python wrapper
 
 
 Install dependecies

 ```
 pip install pyfastnoisesimd ompl-thin pathfinding
 ```

# Credit
[@Elijah1111](https://github.com/Elijah1111)
[@JarenPeckham](https://github.com/jarenpeckham)
...
