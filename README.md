# Shape Viewer
By Sam Ehrenstein (ehrensam@cs.unc.edu)

This is my attempt at creating a 2020s-era version of [Shapemonger](https://andrewthall.org/papers/draft7_7Feb05.pdf), a Windows program for visualizing shapes and their geometric surface properties. I mainly built it on top of Numpy, PyVista, and SymPy (for symbolic calculations).

## Capabilities
Present capabilities include:
- display arbitrary functions of the form $z = f(x,y)$
- plot principal directions, normals, and asymptotic directions and curves
- fully configurable plot options

## Installation
In addition to Python 3.10 or later, Shape Viewer requiries these additional librariies:
```
numpy
scipy
matplotlib
pyvista
yaml
sympy
```
All can be installed using `pip`.

## Usage
Shape Viewer creates PyVista plots based on YAML config files. Some example files are included in the `shapes/` directory. For example, in order to plot a [monkey saddle](https://mathworld.wolfram.com/MonkeySaddle.html), the following command is used:

```
$ python viewer.py shapes/monkeysaddle.yaml
```

### Config file structure
A typical config file looks like this:
```
# Monkey saddle
---
equation: x**3 - 3*x*y**2
lims:
  xmin: -1
  xmax: 1
  ymin: -1
  ymax: 1
  steps: 61
normals:
  plot: True
  color: "green"
principal_dirs:
  plot: True
  colors: ["black", "orange"]
  draw_curves: False
asymptotic_dirs:
  plot: False
  colors: ["red", "blue"]
  draw_curves: True
parabolic_curves: True
```
- `equation` is any sympy-parseable expression string, assumed to represent $f(x,y)$ in an equation for a surface of the form $z=f(x,y)$.
- `lims` controls the limits of the plotted surface.
- `normals` controls whether normals are plotted.
- `principal_dirs` and `asymptotic_dirs` control the plotting of principal and asymptotic directions at points, and whether curves are plotted by integrating the fields of principal and asymptotic directions.
- `parabolic_curves` controls whether parabolic curves are drawn on the surface.