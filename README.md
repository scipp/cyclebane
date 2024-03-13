[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![PyPI badge](http://img.shields.io/pypi/v/cyclebane.svg)](https://pypi.python.org/pypi/cyclebane)
[![Anaconda-Server Badge](https://anaconda.org/scipp/cyclebane/badges/version.svg)](https://anaconda.org/scipp/cyclebane)
[![License: BSD 3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

# Cyclebane

## About

Transform directed acyclic graphs using map-reduce and groupby operations

This library is an attempt to merge the concepts of directed acyclic graphs (DAG) with array-like objects such as NumPy arrays, Pandas DataFrames, or Xarray/Scipp DataArrays.
This could be useful for describing tasks graphs, e.g., when a series of tasks is applied to chunks of an array.
These tasks also have an array structure.
After an reduction operation of chunks, the graph loses this structure, i.e., only a subset of the graph's nodes has array structure.
What if we could work with this structure, even though only parts of the graph follows it?
And what if we could use the power of array slicing with named dimensions, or select by label?
This is what Cyclebane tries to do.

Our initial goal is to support:

- `map` operations of a DAG's source nodes over an array-like (https://docs.dask.org/en/latest/high-level-graphs.html).
  Cyclebane will effectively copy all descendants of those nodes, once for each array element.
  Cyclebane will support joint mappings of multiple source nodes by mapping over, e.g., a DataFrame with multiple columns, as well as chaining independent map operations at different source nodes.
  In the latter case this will effectively broadcast at descendant nodes that depend on multiple such source nodes.
- `reduce` operations at descendants of mapped nodes.
  This will add a new node with edges to all copies of the mapped node being reduced.
  Cyclebane will support reducing only individual axes or all axes, similar to Numpy.
- `groupby` operations similar to Pandas and Xarray (albeit more limited).
- Positional and label-based indexing.
  Cyclebane will support selecting branches that were creating during `map` (or `groupby`) operations based on their indices.
  The graph structure will be left untouched, i.e., nodes after a `reduce` operation will be preserved, but fewer edges will lead to the reduce node.

See also Dask's [High Level Graphs](https://docs.dask.org/en/latest/high-level-graphs.html) for a related concept (without the direct support for any such operations).

## Installation

```sh
python -m pip install cyclebane
```
