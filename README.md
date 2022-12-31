# BetterSmarterFaster
Extension of CircleOfLife where decisions are based on utilities in MDP and Neural Network Models are made to predict utilities.

## ENVIRONMENT 
This has the same environment as the Circle of Life. Here also we have the agent, the predator and the prey. The predator transitions with 60% probability of moving to the neighbor which is closest to prey and 40% of the times randomly to one of it's neighboring vertices.

### DEFINITION of Utility in the MDP
For a given state s, let U∗(s) be the minimal expected number of rounds to catch the prey for an optimal agent, assuming movement order as in The Circle of Life.

### Implementations:

#### U - Star
A method to find utilities U∗(s) of every state possible in the environment. With these calculated utilities, the agent can form an action list with the best possible move for that state which takes it to the final desired goal state.

#### Neural Network Model to predict the value of U Star for a state
A model to predict the value of U∗(s) from the state s. Call this model V .

#### Implementation for partial information setting
In this case, we may not know exactly where the prey or predator is. Consider the unknown prey position case - here, the state of the agent may be represented by the position of the agent, the position of the predator, and a vector of probabilities p. Because there are infinite possible belief states, the optimal utility is hard to solve for. In this case, we might estimate the utility of a given state as the expected utility based on where the prey might be. 

U-partial(s-agent,s-predator,p) = Summation over all prey states (probability of prey at that state * U-star(s-agent,s-predator,s-prey).

#### Simulation of an agent based on Upartial
An agent based on Upartial, in the partial prey info environment case from 'The Circle of Life', using the values of U∗ from above.

#### Neural Network Model for the Partial Information Environment.
A model Vpartial to predict the value Upartial for these partial information states. The training data states( including belief states ) are used to train this Neural Network Model. 
