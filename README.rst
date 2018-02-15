MaStream
========

Clustering Data Streams Using Mass Estimation
=============================================

Usage
-----

Construct a MaStream instance specifying:

* number of h:d-Trees in the ensemble (tree_no), 
* number of data entries used for constructing each tree (tree_train_size), 
* maximum tree depth (max_lvl), 
* stream speed as number of data instances per time unit (horizon)

::

  from mastream.MaStream import MaStream
  mastream = MaStream(tree_no = 10, tree_train_size = 45, max_lvl = 10, horizon = 1000)

Consume a stream:

::

  for idx, entry in enumerate(stream):
        mastream.parse_entry(idx, entry)

After each time unit, the identified labels can be retrieved via:

::

  mastream.get_labels



References
----------

* `Clustering Data Streams Using Mass Estimation <http://ieeexplore.ieee.org/abstract/document/6821162/>`_

  Sabau, Andrei Sorin.
  Symbolic and Numeric Algorithms for Scientific Computing (SYNASC), 2013 15th International Symposium on. IEEE, 2013.