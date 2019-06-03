# Usage 
  Determine symmetry of nanoparticle assembly, especially 4 fold or 6 fold. Analyze the size of nanoparticles.
  NOTE: The current master branch is now Python 2 only. 

# Intro 
* Run sm_46.py to get symmetry map
* Run structure_metric.py to get complete analysis

# General logic 
1. Read file 
1. Get image scale 
1. Get particle centers/radii and binary of the image 
1. Voronoi cells
1. MSM structure to get the assembly orientation
1. Plot symmetry map

# Key parameters
1. Min_feature_size (minum feature size)
1. Morphology (Detect particle with morphology filter)
1. Small (Remove small patches)
1. Areafilter (remove weird voronoi cell)
1. Edge (plot edge cell or not)

# Optimization steps 
First: Tune the min_feature_size to match the particle size 
 
Second: Suggested to use morphology

Third: If weird voronoi cell appears, use areafilter 

Forth: Play with small and edge 
 
 


