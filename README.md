# Self-Organizing Map

In this repository there is an implementation of Self-Organizing Map.

## Self-Organizing Map Class

### Parameters:
- n (int): Number of rows in the SOM network.
- m (int): Number of columns in the SOM network.
- eta (float): Learn rate.
- tau (int): Learning rate decay factor.
- sigma (int): Initial neighborhood radius.
- sigma_dec (int): Neighborhood radius decrease factor.
  
### Methods_
- entrenar (fit).
  Parameters:
    - X (array[array[float]]): Training input data.
    - tam (tuple): Input data dimension.
    - max_iteraciones (int): Maximum number of iterations.
- crear_mapa_dist (create a map with the distences between neurons).
  Returns:
    - Matrix: Map of distances between neurons.
  
### Requirements:
- Python.
- numpy module.
 
### How to install this module
First, clone this repository with the git command below:
```
git clone https://github.com/PanquecaFuriosa/Self-Organizing-Map
```
