### A simple meteo model for France

This is the first iteration of a simple meteo model for France. 


### Big idea 

### The model

Here is a global overview of the architecture :

![Meteo model](images/model.png)

The key idea is to be able to forecast the weather in France with input only coming from the weather stations and radar stations.
The goal of this is to simply provide a weather forecast for the next few hours.

## The data

The data is coming from two sources :
- The weather stations
- The radar stations
- Other data sources (like topology data, ground type data etc.) : IGN and other

Here is a global overview of the data :

![Data](images/data_type.png)


## First step 

1. The first element / step is to forecast the evolution of the radar image.
DONE

2. Second element is to add the ground station input / output to enrich the feature space
DONE

3. Adding the elevation information to improve forecast


4. Adding ground clear sky irradiance
TO BE DONE


4. Extend prediction to whole europe with new dataset !
TO BE DONE

## Current test setup

First test on MF (meteo france) radar : not conclusive too much noise
(scripts/training_grid.py)

Second test on UK dataset : better learning but using UNET / segformer give a oversmoothing image ...
(scripts/training_grid_uk.py)

Third test on UK dataset : using DiT architecture we achieve a better diffusion result (and indeed we get better image)

Forth test on UK dataset : using DiT architecture with local modifycation to reduce compute complexity
