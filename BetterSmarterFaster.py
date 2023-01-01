import itertools
import math
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing as pre
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import csv
import networkx as nx

probability_table=[]                                                        # A list to store the prey probabilities 
reward_vec = []                                                     
utility = {}                                                                # Dictionary to store the utility of a state
reward = {}                                                                 # Dictionary to store the reward of a state
trans_probab = {}                                                           # Dictionary to store the probability of all state changes
step_check=[]
step_check_model=[]
no_of_nodes=50                                                              # Number of nodes of the graph
temp=1

v_partial_weight_dict = {}
weight_dict = {}
Partial_Dataset = []    

def convert(adjacency_list):                                                 # Function to convert a adjacency list into an adjacency matrix
    matrix = np.array(np.random.choice([0, 1], size=(51,51), p=[1, 0]))  
    for i in range(1,51):
        for j in adjacency_list[i]:
            matrix[i][j] = 1
    return matrix        

def modelV():


    print("Data Processing")
    dataset = pd.read_csv("output.csv")

    dataset.columns = ["Agent_Pos", "Prey_Pos","Pred_Pos", "Prey_Dist","Pred_Dist","Utility"]

    x = dataset[["Agent_Pos", "Prey_Pos","Pred_Pos", "Prey_Dist","Pred_Dist"]]
    y = dataset[["Utility"]]

    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= None)
    x_train = x.to_numpy()
    y_train = y.to_numpy()
    print("Type X_train : ", type(x_train))

    print('X_Train shape : ', x_train.shape)
    print('Y_Train shape : ', y_train.shape)
    

    # Hyperbolic Tangent Activation function
    def hyperbolic_tanh(x):
        #return 1/(1 + np.exp(-x))
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    # Hyperbolic derivative
    def derivative_hyperbolic(x):
        #oj = hyperbolic_tanh(x)
        #return oj * (1-oj)
        return 1 - hyperbolic_tanh(x) * hyperbolic_tanh(x)

    # Linear Activation Function
    def linear_activation(x):
        return x

    # Linear derivative
    def derivative_linear(x):
        return 1

    
    # Setting Hyperparameters
    output_sz = y_train.size
    print("Actual out size : ", output_sz)
    np.random.seed(10)
    no_of_features = 5
    no_of_node = 6
    outp_node = 1

    print("5 weights for hidden layers")
    weight_la1 = np.random.randn(no_of_features, no_of_node)
    weight_la2 = np.random.randn(no_of_node, no_of_node)
    weight_la3 = np.random.randn(no_of_node, no_of_node)
    weight_la4 = np.random.randn(no_of_node, no_of_node)
    weight_la5 = np.random.randn(no_of_node, no_of_node)
    output_weight = np.random.randn(no_of_node, outp_node)

    print("weight_la1 : ", weight_la1)
    print(weight_la1.shape)

    rmse_list = []

    epochs = 1000
    eta = 0.001
    alpha = 0.003

    print("Training")

    while epochs > 0:
        # Feedforward for 4 hidden layers by calling activation function
        layer_1 = np.dot(x_train, weight_la1)
        layer_1_out = hyperbolic_tanh(layer_1)

        layer_2 = np.dot(layer_1_out, weight_la2)
        layer_2_out = hyperbolic_tanh(layer_2)

        layer_3 = np.dot(layer_2_out, weight_la3)
        layer_3_out = hyperbolic_tanh(layer_3)

        layer_4 = np.dot(layer_3_out, weight_la4)
        layer_4_out = hyperbolic_tanh(layer_4)

        layer_5 = np.dot(layer_4_out, weight_la5)
        layer_5_out = linear_activation(layer_5)

        output = np.dot(layer_5_out, output_weight)
        final_out = linear_activation(output)

        rmse = np.sqrt(np.mean(np.square(final_out - y_train))) /100
        rmse_list.append(rmse)

        # Backpropagation for 4 hidden layers
        final_err = final_out - y_train
        #final_tanh_derivative = final_err * derivative_hyperbolic(final_out)
        final_lin_derivative = final_err * derivative_linear(final_out)

        layer_5_err = np.dot(final_lin_derivative, output_weight.T)
        layer_5_derivative = layer_5_err * derivative_linear(layer_5_out)


        layer_4_err = np.dot(layer_5_derivative, weight_la5.T)
        layer_4_derivative = layer_4_err * derivative_hyperbolic(layer_4_out)

        layer_3_err = np.dot(layer_4_derivative, weight_la4.T)
        layer_3_derivative = layer_3_err * derivative_hyperbolic(layer_3_out)

        layer_2_err = np.dot(layer_3_derivative, weight_la3.T)
        layer_2_derivative = layer_2_err * derivative_hyperbolic(layer_2_out)

        layer_1_err = np.dot(layer_2_derivative, weight_la2.T)
        layer_1_derivative = layer_1_err * derivative_hyperbolic(layer_1_out)

        # Divide weights as per size of output
        output_weights = np.dot(layer_5_out.T, final_lin_derivative) / output_sz
        weights5 = np.dot(layer_4_out.T, layer_5_derivative) / output_sz
        weights4 = np.dot(layer_3_out.T, layer_4_derivative) / output_sz
        weights3 = np.dot(layer_2_out.T, layer_3_derivative) / output_sz
        weights2 = np.dot(layer_1_out.T, layer_2_derivative) / output_sz
        weights1 = np.dot(x_train.T, layer_1_derivative) / output_sz
        #print("Output Wts : ", output_weights)

        output_weight -=  eta * alpha * output_weights
        weight_la5 -=  eta * alpha * weights5
        weight_la4 -=  eta * alpha * weights4
        weight_la3 -=  eta * alpha * weights3
        weight_la2 -=  eta * alpha * weights2
        weight_la1 -=  eta * alpha * weights1

        epochs -=1

    print("OutPut wts shape: ", output_weights.shape)
    print("Outputs : ", output_weights)

    print("output_weight Shape: ", output_weight.shape)
    print("output_weight : ", output_weight)


    print("Training RMSE: "+str(round(rmse_list[-1],6)))
    print("Model Accuracy : ", 100-round(rmse_list[-1],6))

    print("RMSE Curve")
    plt.title("RMSE Curve")
    plt.ylabel("RMSE")
    plt.xlabel("Epochs")
    plt.plot(rmse_list)
    plt.show()

    weight_dict.clear()
    weight_dict["output_weight"] = output_weight
    weight_dict["weight_la1"] = weight_la1
    weight_dict["weight_la2"] = weight_la2
    weight_dict["weight_la3"] = weight_la3
    weight_dict["weight_la4"] = weight_la4
    weight_dict["weight_la5"] = weight_la5

def get_util_from_V(agentPos,preyPos,predPos,preyDis,predDis):

    print("Predicted Values")
    var = pd.Series([agentPos,preyPos,predPos,preyDis,predDis], index = ["Agent_Pos", "Prey_Pos","Pred_Pos", "Prey_Dist","Pred_Dist"])
    # print("va2 : ", va2)

    #Accessing weights from model
    output_weight = weight_dict["output_weight"]
    weight_la1 = weight_dict["weight_la1"] 
    weight_la2 = weight_dict["weight_la2"]
    weight_la3 = weight_dict["weight_la3"]
    weight_la4 = weight_dict["weight_la4"]
    weight_la5 = weight_dict["weight_la5"]


    # Hyperbolic Tangent Activation function
    def hyperbolic_tanh(x):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    # Hyperbolic derivative
    def derivative_hyperbolic(x):
        return 1 - hyperbolic_tanh(x) * hyperbolic_tanh(x)

    # Linear Activation Function
    def linear_activation(x):
        return x

    # Linear derivative
    def derivative_linear(x):
        return 1


    layer_1 = np.dot(var, weight_la1)
    layer_1_out = hyperbolic_tanh(layer_1)
    #print("layer_1 : ", layer_1)

    layer_2 = np.dot(layer_1_out, weight_la2)
    layer_2_out = hyperbolic_tanh(layer_2)

    layer_3 = np.dot(layer_2_out, weight_la3)
    layer_3_out = hyperbolic_tanh(layer_3)

    layer_4 = np.dot(layer_3_out, weight_la4)
    layer_4_out = hyperbolic_tanh(layer_4)

    layer_5 = np.dot(layer_4_out, weight_la5)
    layer_5_out = linear_activation(layer_5)

    output = np.dot(layer_5_out, output_weight)
    final_out = linear_activation(output)

    print("X-Val : ", var)
    #print("Actual Val : 5.204084483")
    print("Predicted Value : ", final_out)
    return final_out

def probability_distribution2(graph):
    # Previous Probabilities
    copy_prob_table = [None] * len(probability_table)
    for i in range(len(probability_table)):
        copy_prob_table[i] = probability_table[i]
    
    #Calculating and populating next probabilities after the prey has moved
    for i in range(len(probability_table)):
        prob_res_i = 0
        for j in range(len(probability_table)):
            if((i+1) in graph[j+1] or (i+1) == (j+1)):
                fact2 = 1/(len(graph[j+1]) + 1)
            else:
                fact2 = 0
            prob_res_i += copy_prob_table[j] * fact2

        probability_table[i] = prob_res_i

def Survey(index,prey):
    if(index==prey):
        return 1
    else:
        return 0

class Prey:
  def __init__(self, position, parent, distance):
    self.position = position
    self.distance = distance
    self.parent = parent

class Predator:
  def __init__(self, position, parent, distance_Agent):
    self.position = position
    self.distance_Agent = distance_Agent
    self.parent = parent 

class Agent:
  def __init__(self, position, parent, distance_Pred, distance_Prey):
    self.position = position
    self.distance_Pred = distance_Pred
    self.distance_Prey = distance_Prey
    self.parent = parent       

class GraphVisualization:                                                   # Graph visualization using an adjacency matrix
    def __init__(self):
        self.visual = []

    def addEdge(self, a, b):
        temp = [a, b]
        self.visual.append(temp)
    
    def visualize(self,max_state):
        G = nx.Graph()
        color_map = []
        G.add_edges_from(self.visual)
        for node in G:
            if(node == max_state[0]):
                color_map.append('blue')
            elif(node == max_state[1]):
                color_map.append('green')
            elif(node == max_state[2]):
                color_map.append('red')  
            else:
                color_map.append('yellow')           
        nx.draw_networkx(G,node_color=color_map, with_labels=True)
        plt.show()

def utility_initializer(adjacency_list):
    for agent in range(1,no_of_nodes+1):
        for prey in range(1,no_of_nodes+1):
            for pred in range(1,no_of_nodes+1):
                if(agent==pred):
                    utility[agent,prey,pred] = math.inf                         # Terminal state representing a predator win
                else:
                    if(agent==prey):                                            # Terminal state representing an agent win
                        utility[agent,prey,pred] = 0
                    elif(djikstra(adjacency_list,agent,prey)==1):               # States where the agent is a step away from the prey
                        utility[agent,prey,pred] = 1 
                    elif(djikstra(adjacency_list,agent,pred)==1):               # States where the agent is a step away from the pred
                        utility[agent,prey,pred] = 10        
                    else:
                        utility[agent,prey,pred] = 7                            # Utility of every other state

def djikstra(graph,start,end):
    unvisited = {}
    curVertex = start
    for i in range(1,no_of_nodes+1):
        unvisited[i] = float('inf')

    unvisited[start] = 0
    visited = {}

    curVertex = min(unvisited, key=unvisited.get)

    while unvisited:

        curVertex = min(unvisited, key=unvisited.get)
        visited[curVertex] = unvisited[curVertex]

        if curVertex == end:
            return visited[end]

        for nbr in graph.get(curVertex):
            if nbr in visited:
                continue
            tempDist = unvisited[curVertex] + 1
            if(tempDist < unvisited[nbr]):
                unvisited[nbr] = tempDist

        unvisited.pop(curVertex)

def reward_vector():
    for agent in range(1,no_of_nodes+1):
        for prey in range(1,no_of_nodes+1):
            for pred in range(1,no_of_nodes+1):
                if(agent==pred):
                    reward[agent,prey,pred] = 1                                 # Reward for states representing a predator win
                if(agent==prey and agent!=pred):
                    reward[agent,prey,pred] = -1                                # Reward for states representing an agent win
                else:
                    reward[agent,prey,pred] = 0                                 # Reward for every other state

def transition_probab(adjacency_list):
    for key,val in utility.items():
        [agent,prey,pred] = key
        # print("Agent Pos : ", agent)
        # print("Prey Pos: ", prey)
        # print("Pred Pos :", pred)

        agent_nbrs = []
        for x in adjacency_list[agent] :
            agent_nbrs.append(x)
        
        prey_nbrs = []
        for x in adjacency_list[prey] :
            prey_nbrs.append(x)
        
        pred_nbrs = []
        for x in adjacency_list[pred] :
            pred_nbrs.append(x)

        no_of_possib = len(agent_nbrs) * len(pred_nbrs) * len(pred_nbrs)
        shortest_dist_nbr_pred = []
        short_dist = 9999999
        for n in pred_nbrs:
            temp_pred_dist = djikstra(adjacency_list,n,agent)
            if(temp_pred_dist <= short_dist):
                short_dist = temp_pred_dist
                shortest_dist_nbr_pred.append(n)

        for i in agent_nbrs:
            for j in prey_nbrs:
                for k in pred_nbrs:
                    if(k in shortest_dist_nbr_pred):
                        pr = 1/(len(shortest_dist_nbr_pred))*(0.6 + (0.4*(1/len(pred_nbrs))))
                    else:
                        pr = 0.4 * (1/len(pred_nbrs))


                    trans_probab[(agent,prey,pred),(i,j,k)] = 1 * (1/(len(prey_nbrs) + 1)) * (pr)
       
def utility_star_1(adjacency_list):
    reward_vector()                                                             # Initialization of reward vector
    count = 0
    while(True):
        copy_utility={}
        for agent in range(1,no_of_nodes+1):
            for prey in range(1,no_of_nodes+1):
                for pred in range(1,no_of_nodes+1):
                    copy_utility[agent,prey,pred]=utility[agent,prey,pred]
        
        for agent in range(1,no_of_nodes+1):                                        # Execution of value iteration
            nbrs = []
            for x in adjacency_list[agent]:
                nbrs.append(x)
            # print("nbrs ",nbrs)    
            for prey in range(1,no_of_nodes+1):
                for pred in range(1,no_of_nodes+1): 
                    if(agent==prey):
                        continue
                    if(agent==pred):
                        continue
                    if(djikstra(adjacency_list,agent,prey)==1):
                        continue
                    if(djikstra(adjacency_list,agent,pred)==1):
                        continue
                    else:
                        min_val = math.inf              
                        for x in nbrs:                  
                            summation = 2
                            cur_utility = 0
                            for temp_prey in adjacency_list[prey]:    
                                for temp_pred in adjacency_list[pred]:   
                                    transition_probability = trans_probab[(agent,prey,pred),(x,temp_prey,temp_pred)] 
                                    summation = summation + (transition_probability * copy_utility[x,temp_prey,temp_pred])  
                            cur_utility = reward[agent,prey,pred] + (1*summation)
                            if(cur_utility<min_val):
                                min_val=cur_utility 
                        utility[agent,prey,pred] = min_val
                        print("utility[",agent,prey,pred,"]",utility[agent,prey,pred])
        flag=0
        for agent in range(1,no_of_nodes+1):                                            # Convergence check
            for prey in range(1,no_of_nodes+1):
                for pred in range(1,no_of_nodes+1):
                    #print(abs(utility[agent,prey,pred]-copy_utility[agent,prey,pred]))
                    if(abs(utility[agent,prey,pred]-copy_utility[agent,prey,pred])>0.0001):
                        flag=1
                        break
        if(flag==0):
            break      

def Agent1(adjacency_list):
    index=list(range(1,no_of_nodes+1))                                      
    Agent_pos = np.random.choice(index)
    index_pred=list(range(1,no_of_nodes+1))
    index_pred.remove(Agent_pos)
    predator_pos = np.random.choice(index_pred)
    prey_pos = np.random.choice(index_pred)
    Agent1 = Agent(Agent_pos,0,0,0)                                                            # Spawning the agent predator and prey
    Predator1 = Predator(predator_pos,0,0)
    Prey1 = Prey(prey_pos,0,0)

    curState = [Agent1.position,Prey1.position,Predator1.position]

    beta = 1
    count=0
    flag=0
    while(True):
        print("Agent:",Agent1.position," Prey:",Prey1.position," Predator:",Predator1.position)
        count = count+1
        if(count==500):
            break
        nbrs = []
        for x in adjacency_list[Agent1.position]:
            nbrs.append(x)

        min_nbr = 10000
        min_ut = 990
        utility_set=set()
        n_flag=0           
        for x in nbrs:                                                              
            print("utility[x,Prey1.position,Predator1.position] ",x,utility[x,Prey1.position,Predator1.position]," Actual distance ",djikstra(adjacency_list,x,Prey1.position))
            utility_set.add(utility[x,Prey1.position,Predator1.position])
            if(utility[x,Prey1.position,Predator1.position] < min_ut):                        # Finding the neighbor with the lowest utility
                min_ut = utility[x,Prey1.position,Predator1.position]
                min_nbr = x
                n_flag = 1
        print("utility_set ",utility_set)

        if(n_flag==1):
            Agent1.position = min_nbr
        else:
            Agent1.position = nbrs[0]   
        print("Agent1.position ",Agent1.position)

        #Check if Agent Won:
        if(Agent1.position == Prey1.position):
            flag=1
            print("Agent won")
            break
        #Check if Predator Won
        if(Agent1.position == Predator1.position):
            flag=2
            print("Predator won")
            break
        
        # Moving the prey
        l_prey=[]
        for j in adjacency_list[Prey1.position]:
            l_prey.append(j)
        l_prey.append(Prey1.position)    
        prey_next_point=np.random.choice(l_prey)
        Prey1.position=prey_next_point 

        if(Agent1.position == Prey1.position):
            flag=1
            print("Agent won")
            break

        #Move distracted predator
        pred_choice=np.random.choice([1,0],p=[0.6,0.4])
        if(pred_choice==1):
            min_dist_agent=100000
            for i in adjacency_list[Predator1.position]:
                temp=djikstra(adjacency_list,i,Agent1.position)
                print("Predator1 temp ",temp," i ", i)
                if(temp<min_dist_agent):
                    pred_next_pos=i
                    min_dist_agent=temp
            Predator1.position=pred_next_pos  
        elif(pred_choice==0):
            Predator1.position=np.random.choice(adjacency_list[Predator1.position])

        #Check if Predator Won
        if(Agent1.position == Predator1.position):
            flag=2
            print("Predator won")
            break
        print()
    print("count ",count) 
    step_check.append(count)   
    return flag

def Agent_Model(adjacency_list):
    index=list(range(1,no_of_nodes+1))
    Agent_pos = np.random.choice(index)
    index_pred=list(range(1,no_of_nodes+1))
    index_pred.remove(Agent_pos)
    predator_pos = np.random.choice(index_pred)
    prey_pos = np.random.choice(index_pred)
    Agent1 = Agent(Agent_pos,0,0,0)                                                 # Spawning the agent predator and prey
    Predator1 = Predator(predator_pos,0,0)
    Prey1 = Prey(prey_pos,0,0)

    curState = [Agent1.position,Prey1.position,Predator1.position]

    beta = 1
    count=0
    flag=0
    while(True):
        print("Agent:",Agent1.position," Prey:",Prey1.position," Predator:",Predator1.position)
        count = count+1
        if(count==2000):
            break
        nbrs = []
        for x in adjacency_list[Agent1.position]:
            nbrs.append(x)

        min_nbr = 10000
        min_ut = 990
        utility_set=set()
        for x in nbrs:                                                        # Finding the lowest neightbor with the lowest value of predicted utility
            if(utility[x,Prey1.position,Predator1.position]==math.inf or utility[x,Prey1.position,Predator1.position]==0):
                predicted_utility = utility[x,Prey1.position,Predator1.position]
            else:    
                predicted_utility = get_util_from_V(x,Prey1.position,Predator1.position,djikstra(adjacency_list,x,Prey1.position),djikstra(adjacency_list,x,Predator1.position))
                predicted_utility = predicted_utility[0]
            print("utility[x,Prey1.position,Predator1.position] ",x,predicted_utility," Actual distance ",djikstra(adjacency_list,x,Prey1.position))
            utility_set.add(predicted_utility)
            if(predicted_utility < min_ut):
                min_ut = predicted_utility
                min_nbr = x
        print("utility_set ",utility_set)
        Agent1.position = min_nbr

        print("Agent1.position ",Agent1.position)

        #Check if Agent Won:
        if(Agent1.position == Prey1.position):
            flag=1
            print("Agent won")
            break
        #Check if Predator Won
        if(Agent1.position == Predator1.position):
            flag=2
            print("Predator won")
            break
        
        # Moving the prey

        l_prey=[]
        for j in adjacency_list[Prey1.position]:
            l_prey.append(j)
        l_prey.append(Prey1.position)    
        prey_next_point=np.random.choice(l_prey)
        Prey1.position=prey_next_point 

        if(Agent1.position == Prey1.position):
            flag=1
            print("Agent won")
            break

        #Move predator
        pred_choice=np.random.choice([1,0],p=[0.6,0.4])
        #pred_choice=0
        if(pred_choice==1):
            min_dist_agent=100000
            for i in adjacency_list[Predator1.position]:
                temp=djikstra(adjacency_list,i,Agent1.position)
                print("Predator1 temp ",temp," i ", i)
                if(temp<min_dist_agent):
                    pred_next_pos=i
                    min_dist_agent=temp
            Predator1.position=pred_next_pos  
        elif(pred_choice==0):
            Predator1.position=np.random.choice(adjacency_list[Predator1.position])

        #Check if Predator Won
        if(Agent1.position == Predator1.position):
            flag=2
            print("Predator won")
            break
        print()
    print("count ",count) 
    step_check_model.append(count)   
    return flag

def Upartial_Agent(adjacency_list):
    count_prey_pos = 0
    index=list(range(1,no_of_nodes+1))
    Agent_pos = np.random.choice(index)
    index_pred=list(range(1,no_of_nodes+1))
    index_pred.remove(Agent_pos)
    predator_pos = np.random.choice(index_pred)
    prey_pos = np.random.choice(index_pred)

    Agent1 = Agent(Agent_pos,0,0,0)
    Predator1 = Predator(predator_pos,0,0)
    Prey1 = Prey(prey_pos,0,0)

    #Clearing probability table
    probability_table.clear()
    #Inserting Raw probabilities
    for x in range(no_of_nodes):
        probability_table.append(1/no_of_nodes)
    
    probability_table[Agent1.position-1]=0

    for y in range(no_of_nodes):
        if(y != Agent1.position-1):
            probability_table[y]=(1/(no_of_nodes-1))

    #Initial Survey
    surveyList = list(range(1,no_of_nodes+1))
    surveyList.remove(Agent_pos)
    surveyNode = np.random.choice(surveyList)

    surveyRes = Survey(surveyNode, Prey1.position)

    loopCounter = 0
    flag = 0
    while(True):

        loopCounter += 1
        
        #Update Probabilities after survey
        if(surveyRes == 1):
            #Prey Found -> Update Probab table
            for i in range(len(probability_table)):
                probability_table[i] = 0
            probability_table[surveyNode-1] = 1

            agent_nbrs = []
            for x in adjacency_list[Agent1.position]:
                agent_nbrs.append(x)

            upartial_dict = {}
            upartial_dict.clear()
            for nbr in agent_nbrs:                
                
                summatn = 0
                l=[]
                flag_inf=0
                for probab_index in range(len(probability_table)):
                    if(utility[nbr,probab_index + 1,Predator1.position] in [math.inf]):
                        flag_inf=1
                        break
                    summatn += probability_table[probab_index] * utility[nbr,probab_index + 1,Predator1.position]
                if(flag_inf==0):
                    upartial_dict[nbr,Predator1.position] = summatn
                    #VISH
                    l.append(nbr)
                    l.append(Predator1.position)
                    for a in range(len(probability_table)):
                        l.append(probability_table[a])
                    l.append(upartial_dict[nbr,Predator1.position])
                    Partial_Dataset.append(l)
                #VISH
            #Upartial values calculated -> Now move the agent to the lowest Upartial position
            minUtil = 90000
            for key, val in upartial_dict.items():
                if(upartial_dict[key] < minUtil):
                    [ag,pr] = key
                    mn_nbr = ag

            Agent1.position = mn_nbr

            #No need to update probability table After Agent Move

            #Check if Agent Won or Died
            if(Agent1.position == Prey1.position):
                flag = 1
                print("Agent Won")
                break
            # if(Agent1.position == Predator1.position):
            #     flag = 2
            #     print("Predator Won")
            #     break

            #Move Prey
            l_prey=[]
            for j in adjacency_list[Prey1.position]:
                l_prey.append(j)
            l_prey.append(Prey1.position)    
            prey_next_point=np.random.choice(l_prey)
            Prey1.position=prey_next_point

            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break

            #Update belief Probablity Table of Agent
            probability_distribution2(adjacency_list)

            #Move predator
            pred_choice=np.random.choice([1,0],p=[0.6,0.4])
            #pred_choice=0
            if(pred_choice==1):
                min_dist_agent=100000
                for i in adjacency_list[Predator1.position]:
                    temp=djikstra(adjacency_list,i,Agent1.position)
                    print("Predator1 temp ",temp," i ", i)
                    if(temp<min_dist_agent):
                        pred_next_pos=i
                        min_dist_agent=temp
                Predator1.position=pred_next_pos  
            elif(pred_choice==0):
                Predator1.position=np.random.choice(adjacency_list[Predator1.position])

            #Check if predator won
            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break

        if(surveyRes == 0):
            #Update probabilities
            for i in range(len(probability_table)):
                probability_table[i] = 1/(no_of_nodes-2)
            probability_table[surveyNode-1] = 0
            probability_table[Agent1.position-1] = 0

            agent_nbrs = []
            for x in adjacency_list[Agent1.position]:
                agent_nbrs.append(x)

            upartial_dict = {}
            upartial_dict.clear()
            for nbr in agent_nbrs:                
                
                summatn = 0
                l=[]
                flag_inf=0
                for probab_index in range(len(probability_table)):
                    if(utility[nbr,probab_index + 1,Predator1.position] in [math.inf]):
                        flag_inf=1
                        break
                    summatn += probability_table[probab_index] * utility[nbr,probab_index + 1,Predator1.position]
                if(flag_inf==0):
                    upartial_dict[nbr,Predator1.position] = summatn
                    #VISH
                    l.append(nbr)
                    l.append(Predator1.position)
                    for a in range(len(probability_table)):
                        l.append(probability_table[a])
                    l.append(upartial_dict[nbr,Predator1.position])
                    Partial_Dataset.append(l)
                    #VISH

            #Upartial values calculated -> Now move the agent to the lowest Upartial position
            minUtil = 90000
            mn_nbr = np.random.choice(adjacency_list[Agent1.position])
            for key, val in upartial_dict.items():
                if(upartial_dict[key] < minUtil):
                    [ag,pr] = key
                    mn_nbr = ag

            Agent1.position = mn_nbr

            #Check if agent won or died
            if(Agent1.position == Prey1.position):
                flag = 1
                print("Agent Won")
                break
            # if(Agent1.position == Predator1.position):
            #     flag = 2
            #     print("Predator Won")
            #     break

            #Move Prey
            l_prey=[]
            for j in adjacency_list[Prey1.position]:
                l_prey.append(j)
            l_prey.append(Prey1.position)    
            prey_next_point=np.random.choice(l_prey)
            Prey1.position=prey_next_point

            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break

            #Update belief Probablity Table of Agent
            probability_distribution2(adjacency_list)

            #Move predator
            pred_choice=np.random.choice([1,0],p=[0.6,0.4])
            #pred_choice=0
            if(pred_choice==1):
                min_dist_agent=100000
                for i in adjacency_list[Predator1.position]:
                    temp=djikstra(adjacency_list,i,Agent1.position)
                    print("Predator1 temp ",temp," i ", i)
                    if(temp<min_dist_agent):
                        pred_next_pos=i
                        min_dist_agent=temp
                Predator1.position=pred_next_pos  
            elif(pred_choice==0):
                Predator1.position=np.random.choice(adjacency_list[Predator1.position])

            #Check if predator won
            if(Agent1.position == Predator1.position):
                flag=2
                print("Predator Won")
                break

        # Take the max of the probability table for position of the new survey
        maxProbab = max(probability_table)
        tempL = []
        for f in range(len(probability_table)):
            if(probability_table[f] == maxProbab):
                tempL.append(f)
        
        maxPosition = np.random.choice(tempL) + 1
        surveyNode = maxPosition
        surveyRes = Survey(surveyNode, Prey1.position)

    if(flag==1):
        return 1
    if(flag==2):
        return 2
    if(flag==0):
        return 0


def modelV_Partial():


    print("Data Processing")
    dataset = shuffle(pd.read_csv("output_partial.csv"))

    dataset.columns = ["Agent_Pos", "Pred_Pos","Belief1", "Belief2","Belief3","Belief4","Belief5",
    "Belief6","Belief7","Belief8","Belief9","Belief10","Belief11","Belief12","Belief13","Belief14",
    "Belief15","Belief16","Belief17","Belief18","Belief19","Belief20","Belief21","Belief22","Belief23",
    "Belief24","Belief25","Belief26","Belief27","Belief28","Belief29","Belief30","Belief31","Belief32",
    "Belief33","Belief34","Belief35","Belief36","Belief37","Belief38","Belief39","Belief40","Belief41",
    "Belief42","Belief43","Belief44","Belief45","Belief46","Belief47","Belief48","Belief49","Belief50","Utility"]

    

    x = dataset[["Agent_Pos", "Pred_Pos","Belief1", "Belief2","Belief3","Belief4","Belief5",
    "Belief6","Belief7","Belief8","Belief9","Belief10","Belief11","Belief12","Belief13","Belief14",
    "Belief15","Belief16","Belief17","Belief18","Belief19","Belief20","Belief21","Belief22","Belief23",
    "Belief24","Belief25","Belief26","Belief27","Belief28","Belief29","Belief30","Belief31","Belief32",
    "Belief33","Belief34","Belief35","Belief36","Belief37","Belief38","Belief39","Belief40","Belief41",
    "Belief42","Belief43","Belief44","Belief45","Belief46","Belief47","Belief48","Belief49","Belief50"]]
    y = dataset[["Utility"]]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2)

    x_train = pre.StandardScaler().fit_transform(x_train)
    x_test = pre.StandardScaler().fit_transform(x_test)

    #x_train = x.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    print("Type X_train : ", type(x_train))

    print('X_Train shape : ', x_train.shape)
    print('Y_Train shape : ', y_train.shape)
    # print('X_test shape: ', x_test.shape)
    # print('Y_test shape : ', y_test.shape)

    # Convert pd dataframe to numpy array
    #y_train = y_train.to_numpy()
    #y_test = y_test.to_numpy()

    # Hyperbolic Tangent Activation function
    def hyperbolic_tanh(x):
        #return 1/(1 + np.exp(-x))
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    # Hyperbolic derivative
    def derivative_hyperbolic(x):
        #oj = hyperbolic_tanh(x)
        #return oj * (1-oj)
        return 1 - hyperbolic_tanh(x) * hyperbolic_tanh(x)

    # Linear Activation Function
    def linear_activation(x):
        return x

    # Linear derivative
    def derivative_linear(x):
        return 1

    
    # Setting Hyperparameters
    output_sz = y_train.size
    print("Actual out size : ", output_sz)
    np.random.seed(10)
    no_of_features = 52
    no_of_node = 8
    outp_node = 1

    print("5 weights for hidden layers")
    weight_la1 = np.random.randn(no_of_features, no_of_node)
    weight_la2 = np.random.randn(no_of_node, no_of_node)
    weight_la3 = np.random.randn(no_of_node, no_of_node)
    weight_la4 = np.random.randn(no_of_node, no_of_node)
    weight_la5 = np.random.randn(no_of_node, no_of_node)
    output_weight = np.random.randn(no_of_node, outp_node)

    print("weight_la1 : ", weight_la1)
    print(weight_la1.shape)

    rmse_list = []
    rmse_tst_list = []

    epochs = 1000
    alpha = 0.00003

    print("Training")

    while epochs > 0:
        # Feedforward for 4 hidden layers by calling activation function
        layer_1 = np.dot(x_train, weight_la1)
        layer_1_out = hyperbolic_tanh(layer_1)
        #for test
        p1 = np.dot(x_test, weight_la1)
        p1_out = hyperbolic_tanh(p1)

        layer_2 = np.dot(layer_1_out, weight_la2)
        layer_2_out = hyperbolic_tanh(layer_2)
        #for test
        p2 = np.dot(p1_out, weight_la2)
        p2_out = hyperbolic_tanh(p2)

        layer_3 = np.dot(layer_2_out, weight_la3)
        layer_3_out = hyperbolic_tanh(layer_3)
        #for test
        p3 = np.dot(p2_out, weight_la3)
        p3_out = hyperbolic_tanh(p3)

        layer_4 = np.dot(layer_3_out, weight_la4)
        layer_4_out = hyperbolic_tanh(layer_4)
        #for test
        p4 = np.dot(p3_out, weight_la4)
        p4_out = hyperbolic_tanh(p4)
        
        layer_5 = np.dot(layer_4_out, weight_la5)
        layer_5_out = linear_activation(layer_5)
        #for test
        p5 = np.dot(p4_out, weight_la5)
        p5_out = linear_activation(p5)

        output = np.dot(layer_5_out, output_weight)
        final_out = linear_activation(output)
        #for test
        p6 = np.dot(p5_out, output_weight)
        p6_out = linear_activation(p6)

        rmse = np.sqrt(np.mean(np.square(final_out - y_train))) /100
        rmse_list.append(rmse)
        #Error with Test
        rmse_tst = np.sqrt(np.mean(np.square(p6_out - y_test))) /100
        rmse_tst_list.append(rmse_tst)

        # Backpropagation for hidden layers
        final_err = final_out - y_train
        #final_tanh_derivative = final_err * derivative_hyperbolic(final_out)
        final_lin_derivative = final_err * derivative_linear(final_out)

        layer_5_err = np.dot(final_lin_derivative, output_weight.T)
        layer_5_derivative = layer_5_err * derivative_linear(layer_5_out)


        layer_4_err = np.dot(layer_5_derivative, weight_la5.T)
        layer_4_derivative = layer_4_err * derivative_hyperbolic(layer_4_out)

        layer_3_err = np.dot(layer_4_derivative, weight_la4.T)
        layer_3_derivative = layer_3_err * derivative_hyperbolic(layer_3_out)

        layer_2_err = np.dot(layer_3_derivative, weight_la3.T)
        layer_2_derivative = layer_2_err * derivative_hyperbolic(layer_2_out)

        layer_1_err = np.dot(layer_2_derivative, weight_la2.T)
        layer_1_derivative = layer_1_err * derivative_hyperbolic(layer_1_out)

        # Divide weights as per size of output
        output_weights = np.dot(layer_5_out.T, final_lin_derivative) / output_sz
        weights5 = np.dot(layer_4_out.T, layer_5_derivative) / output_sz
        weights4 = np.dot(layer_3_out.T, layer_4_derivative) / output_sz
        weights3 = np.dot(layer_2_out.T, layer_3_derivative) / output_sz
        weights2 = np.dot(layer_1_out.T, layer_2_derivative) / output_sz
        weights1 = np.dot(x_train.T, layer_1_derivative) / output_sz
        #print("Output Wts : ", output_weights)

        output_weight -=   alpha * output_weights
        weight_la5 -=   alpha * weights5
        weight_la4 -=   alpha * weights4
        weight_la3 -=   alpha * weights3
        weight_la2 -=   alpha * weights2
        weight_la1 -=   alpha * weights1

        epochs -=1

    print("OutPut wts shape: ", output_weights.shape)
    print("Outputs : ", output_weights)

    print("output_weight Shape: ", output_weight.shape)
    print("output_weight : ", output_weight)


    print("Training RMSE: "+str(round(rmse_list[-1],6)))
    print("Model Accuracy : ", 100-round(rmse_list[-1],6))

    print("RMSE Curve")
    plt.title("RMSE Curve")
    plt.ylabel("RMSE")
    plt.xlabel("Epochs")
    plt.plot(rmse_list )
    plt.plot(rmse_tst_list,'-.')
    plt.show()

    v_partial_weight_dict.clear()
    v_partial_weight_dict["output_weight"] = output_weight
    v_partial_weight_dict["weight_la1"] = weight_la1
    v_partial_weight_dict["weight_la2"] = weight_la2
    v_partial_weight_dict["weight_la3"] = weight_la3
    v_partial_weight_dict["weight_la4"] = weight_la4
    v_partial_weight_dict["weight_la5"] = weight_la5

def get_util_from_V_Partial(agentPos,predPos,probability_table):

    print("Predicted Values")
    var = pd.Series([agentPos,predPos,probability_table[0],probability_table[1],probability_table[2],probability_table[3],
    probability_table[4],probability_table[5],probability_table[6],probability_table[7],probability_table[8],probability_table[9],
    probability_table[10],probability_table[11],probability_table[12],probability_table[13],probability_table[14],probability_table[15],
    probability_table[16],probability_table[17],probability_table[18],probability_table[19],probability_table[20],
    probability_table[21],probability_table[22],probability_table[23],probability_table[24],probability_table[25],
    probability_table[26],probability_table[27],probability_table[28],probability_table[29],probability_table[30],
    probability_table[31],probability_table[32],probability_table[33],probability_table[34],probability_table[35],
    probability_table[36],probability_table[37],probability_table[38],probability_table[39],probability_table[40],
    probability_table[41],probability_table[42],probability_table[43],probability_table[44],probability_table[45],
    probability_table[46],probability_table[47],probability_table[48],probability_table[49]], 
    index = ["Agent_Pos", "Pred_Pos","Belief1", "Belief2","Belief3","Belief4","Belief5",
    "Belief6","Belief7","Belief8","Belief9","Belief10","Belief11","Belief12","Belief13","Belief14",
    "Belief15","Belief16","Belief17","Belief18","Belief19","Belief20","Belief21","Belief22","Belief23",
    "Belief24","Belief25","Belief26","Belief27","Belief28","Belief29","Belief30","Belief31","Belief32",
    "Belief33","Belief34","Belief35","Belief36","Belief37","Belief38","Belief39","Belief40","Belief41",
    "Belief42","Belief43","Belief44","Belief45","Belief46","Belief47","Belief48","Belief49","Belief50"])
    # print("va2 : ", va2)

    #Accessing weights from model
    output_weight = v_partial_weight_dict["output_weight"]
    weight_la1 = v_partial_weight_dict["weight_la1"] 
    weight_la2 = v_partial_weight_dict["weight_la2"]
    weight_la3 = v_partial_weight_dict["weight_la3"]
    weight_la4 = v_partial_weight_dict["weight_la4"]
    weight_la5 = v_partial_weight_dict["weight_la5"]


    # Hyperbolic Tangent Activation function
    def hyperbolic_tanh(x):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    # Hyperbolic derivative
    def derivative_hyperbolic(x):
        return 1 - hyperbolic_tanh(x) * hyperbolic_tanh(x)

    # Linear Activation Function
    def linear_activation(x):
        return x

    # Linear derivative
    def derivative_linear(x):
        return 1


    layer_1 = np.dot(var, weight_la1)
    layer_1_out = hyperbolic_tanh(layer_1)
    #print("layer_1 : ", layer_1)

    layer_2 = np.dot(layer_1_out, weight_la2)
    layer_2_out = hyperbolic_tanh(layer_2)

    layer_3 = np.dot(layer_2_out, weight_la3)
    layer_3_out = hyperbolic_tanh(layer_3)

    layer_4 = np.dot(layer_3_out, weight_la4)
    layer_4_out = hyperbolic_tanh(layer_4)

    layer_5 = np.dot(layer_4_out, weight_la5)
    layer_5_out = linear_activation(layer_5)

    output = np.dot(layer_5_out, output_weight)
    final_out = linear_activation(output)

    print("X-Val : ", var)
    #print("Actual Val : 5.204084483")
    print("Predicted Value : ", final_out)
    return final_out

def VPartial_Agent(adjacency_list):
    index=list(range(1,no_of_nodes+1))
    Agent_pos = np.random.choice(index)
    index_pred=list(range(1,no_of_nodes+1))
    index_pred.remove(Agent_pos)
    predator_pos = np.random.choice(index_pred)
    prey_pos = np.random.choice(index_pred)
    Agent1 = Agent(Agent_pos,0,0,0)
    Predator1 = Predator(predator_pos,0,0)
    Prey1 = Prey(prey_pos,0,0)
    
    #Clearing probability table
    probability_table.clear()
    #Inserting Raw probabilities
    for x in range(no_of_nodes):
        probability_table.append(1/no_of_nodes)
    
    probability_table[Agent1.position-1]=0

    for y in range(no_of_nodes):
        if(y != Agent1.position-1):
            probability_table[y]=(1/(no_of_nodes-1))

    #Initial Survey
    surveyList = list(range(1,no_of_nodes+1))
    surveyList.remove(Agent_pos)
    surveyNode = np.random.choice(surveyList)

    surveyRes = Survey(surveyNode, Prey1.position)

    loopCounter = 0
    flag = 0
    while(True):
        loopCounter +=1
        
        if(surveyRes == 1):
            #Prey Found -> Update Probab table
            for i in range(len(probability_table)):
                probability_table[i] = 0
            probability_table[surveyNode-1] = 1

            agent_nbrs = []
            for x in adjacency_list[Agent1.position]:
                agent_nbrs.append(x)
                
            min_Util = 90000000
            for nbr in agent_nbrs:
                util_from_vPartial = get_util_from_V_Partial(nbr, Predator1.position, probability_table)
                if(util_from_vPartial < min_Util):
                    Agent1.position = nbr
            
            #Check if agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent Won")
                break
            
            #Move Prey
            l_prey=[]
            for j in adjacency_list[Prey1.position]:
                l_prey.append(j)
            l_prey.append(Prey1.position)    
            prey_next_point=np.random.choice(l_prey)
            Prey1.position=prey_next_point

            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break

            #Update belief Probablity Table of Agent
            probability_distribution2(adjacency_list)

            #Move predator
            pred_choice=np.random.choice([1,0],p=[0.6,0.4])
            #pred_choice=0
            if(pred_choice==1):
                min_dist_agent=100000
                for i in adjacency_list[Predator1.position]:
                    temp=djikstra(adjacency_list,i,Agent1.position)
                    print("Predator1 temp ",temp," i ", i)
                    if(temp<min_dist_agent):
                        pred_next_pos=i
                        min_dist_agent=temp
                Predator1.position=pred_next_pos  
            elif(pred_choice==0):
                Predator1.position=np.random.choice(adjacency_list[Predator1.position])
            if(Predator1.position == Agent1.position):
                flag = 2
                print("Predator Won")
                break
            
        if(surveyRes == 0):
            #Update probabilities
            for i in range(len(probability_table)):
                probability_table[i] = 1/(no_of_nodes-2)
            probability_table[surveyNode-1] = 0
            probability_table[Agent1.position-1] = 0

            agent_nbrs = []
            for x in adjacency_list[Agent1.position]:
                agent_nbrs.append(x)

            
            min_Util = 90000000
            for nbr in agent_nbrs:
                util_from_vPartial = get_util_from_V_Partial(nbr, Predator1.position, probability_table)
                if(util_from_vPartial < min_Util):
                    Agent1.position = nbr
            
            #Check if agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent Won")
                break
            
            #Move Prey
            l_prey=[]
            for j in adjacency_list[Prey1.position]:
                l_prey.append(j)
            l_prey.append(Prey1.position)    
            prey_next_point=np.random.choice(l_prey)
            Prey1.position=prey_next_point

            #Check if Agent Won
            if(Agent1.position == Prey1.position):
                flag=1
                print("Agent won")
                break

            #Update belief Probablity Table of Agent
            probability_distribution2(adjacency_list)

            #Move predator
            pred_choice=np.random.choice([1,0],p=[0.6,0.4])
            #pred_choice=0
            if(pred_choice==1):
                min_dist_agent=100000
                for i in adjacency_list[Predator1.position]:
                    temp=djikstra(adjacency_list,i,Agent1.position)
                    print("Predator1 temp ",temp," i ", i)
                    if(temp<min_dist_agent):
                        pred_next_pos=i
                        min_dist_agent=temp
                Predator1.position=pred_next_pos  
            elif(pred_choice==0):
                Predator1.position=np.random.choice(adjacency_list[Predator1.position])
                
            if(Predator1.position == Agent1.position):
                flag=2
                print("Predator Won")
                break
                
        # Take the max of the probability table for position of the new survey
        maxProbab = max(probability_table)
        tempL = []
        for f in range(len(probability_table)):
            if(probability_table[f] == maxProbab):
                tempL.append(f)
        
        maxPosition = np.random.choice(tempL) + 1
        surveyNode = maxPosition
        surveyRes = Survey(surveyNode, Prey1.position)

    if(flag==1):
        return 1
    if(flag==2):
        return 2
    if(flag==0):
        return 0
  
def compare_Vpartial(adjacency_list,agentPos,predPos,probability_table):
    value_of_Vpartial = get_util_from_V_Partial(agentPos, predPos, probability_table)
    
    summatn = 0
    predDis = djikstra(adjacency_list, agentPos, predPos)
    for probab_index in range(len(probability_table)):
        preyDis = djikstra(adjacency_list, agentPos, probab_index + 1)
        summatn += probability_table[probab_index] * get_util_from_V(agentPos, probab_index + 1, predPos, preyDis, predDis)
    value_from_simply_plugging_V = summatn
    diff = abs(value_from_simply_plugging_V - value_of_Vpartial)
    print("Difference : ", diff)
        

count_agent_complete=0
count_pred_complete=0
count_agent_model=0
count_pred_model=0
count_agent_upartial = 0
count_pred_upartial = 0
count_agent_Vpartial = 0
count_pred_Vpartial = 0

tie=0
tie2=0
tie3=0
tie4=0

for x in range(0,1):
    temp=1
    adjacency_list={}
    for i in range(1,no_of_nodes+1):
        if(i==1):
            adjacency_list[temp]=[no_of_nodes,temp+1]
        elif(i==no_of_nodes):    
            adjacency_list[temp]=[temp-1,1]
        else:    
            adjacency_list[temp]=[temp-1,temp+1]
        temp=temp+1
    print(adjacency_list)
    print("----------------------")
    temp=1
    degree_check={}
    for i in range(1,no_of_nodes+1):
        if(i==1):
            degree_check[temp]=[no_of_nodes,temp+1]
        elif(i==no_of_nodes):    
            degree_check[temp]=[temp-1,1]
        else:    
            degree_check[temp]=[temp-1,temp+1]
        temp=temp+1
    index=list(range(1,no_of_nodes+1))
    count=0
    flag=0
    while(degree_check):
        choice = np.random.choice(index)
        l= list(range(choice-5,choice+5+1))
        list_of_possible_edges=[]
        for x in l:
            if(x==0):
                continue
            if(x<1):
                if(no_of_nodes+1+x in index):
                    list_of_possible_edges.append(no_of_nodes+1+x)
                else:
                    continue    
            elif(x>no_of_nodes):
                if(x-no_of_nodes in index):
                    list_of_possible_edges.append(x-no_of_nodes) 
                else:
                    continue    
            else:
                list_of_possible_edges.append(x) 
        list_of_possible_edges = list(dict.fromkeys(list_of_possible_edges))  
        list_of_possible_edges = [ x for x in list_of_possible_edges if len(adjacency_list[x])==2]
        list_of_possible_edges.remove(choice)
        if(choice-1>0 and choice-1 in list_of_possible_edges):
            if(choice==1):
                list_of_possible_edges.remove(no_of_nodes)
            else:
                list_of_possible_edges.remove(choice-1)   
        if(choice+1<=no_of_nodes and choice+1 in list_of_possible_edges):
            if(choice==no_of_nodes):
                list_of_possible_edges.remove(1)
            else:
                list_of_possible_edges.remove(choice+1)    
        #print("list_of_possible_edges ",list_of_possible_edges)
        
        if(len(list_of_possible_edges)==0):
            index.remove(choice)
            del degree_check[choice] 
        else:
            chosen_edge=np.random.choice(list_of_possible_edges)
            #print("chosen_edge ",chosen_edge)
            if(len(adjacency_list[chosen_edge])==2):
                adjacency_list[chosen_edge].append(choice)
                adjacency_list[choice].append(chosen_edge)
                count=count+1
                
                flag=1
        
        if(flag==1):
            index.remove(choice)
            index.remove(chosen_edge)   
            flag=0 
            del degree_check[choice]   
            del degree_check[chosen_edge] 

    print(adjacency_list)
    print(count)
    utility_initializer(adjacency_list)
    transition_probab(adjacency_list)
    utility_star_1(adjacency_list)
    max_u = 0       
    for w in range(0,3000):
        var=Agent1(adjacency_list)
        if(var==1):
            count_agent_complete=count_agent_complete+1
        if(var==2):
            count_pred_complete=count_pred_complete+1  
        if(var==0):
            tie=tie+1
    for key in utility:
        if(utility[key]!= math.inf and utility[key] != 10):   # math.inf
            if(utility[key]>max_u):
                max_u=utility[key]
                max_state = key
    print("Max utility:",max_u," State:",max_state) 

    # MODEL
    X = []
    y = []
    for agent in range(1,51):
        for prey in range(1,51):
            for pred in range(1,51):
                if(utility[agent,prey,pred] not in [math.inf,10,0,1]):
                    l=[]
                    prey_dist=djikstra(adjacency_list,agent,prey)
                    pred_dist=djikstra(adjacency_list,agent,pred)
                    l.append(1)
                    l.append(agent)
                    l.append(prey)
                    l.append(pred)
                    l.append(prey_dist)
                    l.append(pred_dist)
                    X.append(l)
                    y.append(utility[agent,prey,pred])
    with open('output.csv', 'w') as output:
        writer = csv.writer(output)
        count=0
        writer.writerow(["Agent_Pos", "Prey_Pos","Pred_Pos", "Prey_Dist","Pred_Dist","Utility"])
        for i in X:
            writer.writerow([i[1],i[2],i[3],i[4],i[5],utility[i[1],i[2],i[3]]])

    modelV()
    for w in range(0,3000):
        var=Agent_Model(adjacency_list)
        if(var==1):
            count_agent_model=count_agent_model+1
        if(var==2):
            count_pred_model=count_pred_model+1  
        if(var==0):
            tie2=tie2+1
    # MODEL 

    max_u = 0       
    for w in range(0,3000):
        var=Upartial_Agent(adjacency_list)
        if(var==1):
            count_agent_upartial=count_agent_upartial+1
        if(var==2):
            count_pred_upartial=count_pred_upartial+1  
        if(var==0):
            tie3=tie3+1
    # VISH
    with open('output_partial.csv', 'w') as output:
        writer = csv.writer(output)
        count=0
        #writer.writerow(["Agent_Pos", "Pred_Pos","Prey_Probability","Utility"])
        print(Partial_Dataset)
        for f in Partial_Dataset:
            temp=[]
            for j in f:
                temp.append(j)
            writer.writerow(temp)
    
    max_u = 0
    modelV_Partial()     
    for w in range(0,3000):
        var=VPartial_Agent(adjacency_list)
        if(var==1):
            count_agent_Vpartial=count_agent_Vpartial+1
        if(var==2):
            count_pred_Vpartial=count_pred_Vpartial+1  
        if(var==0):
            tie4=tie4+1        
    

adjMatrix = convert(adjacency_list)
#print(adjMatrix)
G = GraphVisualization()

for i in range(1,51):
    for j in range(1,51):
        if(adjMatrix[i][j]==1):
            G.addEdge(i, j)

G.visualize(max_state)


