'''
manu_sandbox.markov_decisions  -  Created on Oct 7, 2013
@author: M. Moussallam

First experiment, suppose I know a transition matrix between algorithm states,
which means I know the probability of selecting atom i after having selected the atom j
Then I observe the projection, and I will select the atom that maximize a mixed criterion
between the projections and the probability of the atom:
- lambda=zero lead to giving no weight to the probabilities
- lambda=infinity lead to giving nop weight to the projections
'''


