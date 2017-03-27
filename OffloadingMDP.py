import numpy as np
import random
import enum
from math import exp
import os
import utility



import mdptoolbox, mdptoolbox.example

#np.random.seed(0)

BIGVALUE = 10000

def params_def_init():
    # This function read parameters from parameters from parameters.ini file
    # and define the corresponding global variables
    utility.log_info("Enter params_def_init()")
    parmfilename = os.path.join(utility.get_current_dir(),"parameters.ini")
    if os.path.isfile(parmfilename) == False:
        log_error("parameters.ini file is not exist!")
        return False
    utility.log_info("------Parameters from ini file------")
    parmfile = open(parmfilename,"r")
    line = parmfile.readline()
    while line:
        stripeline = line.strip()
        if ( stripeline[0:1] == "#"):#skip the comment line with "#" in the head of the line
            line = parmfile.readline()
            continue
        key_value = stripeline.split("#")[0].split("=")
        if ( len(key_value) >= 2 ):
            if key_value[0].strip()[0:6] == "g_list":
                value_list = key_value[1].split(":")
                globals()[key_value[0].strip()] = [float(x)  for x in value_list]
            else:
                globals()[key_value[0].strip()] = float(key_value[1].strip())
            #define the global variables, the value is asummed as float type
            utility.log_info("\t\t"+key_value[0].strip()+"\t=\t"+key_value[1].strip())
        line = parmfile.readline()
    parmfile.close()
    utility.log_info("------------End------------")

def extractlist(prefix):
    prefixlist = list()
    prefixlen = len(prefix)
    for v in globals():
        if v.__str__()[0:prefixlen] == prefix:
            prefixlist.append(globals()[v])
    #prefixlist.sort()
    return prefixlist

#
# This new Environment class considers engergy consumption of mobile user
#
class EnvironmentEX(object):
    #              width(y) ->
    #    (0,0)________________
    #         |___|___|___|___|
    # height  |___|___|___|___|
    #  (x)    |___|___|___|___|
    #         |___|___|___|___|
    #
    #
    #
    #
    #

    def __init__(self, width, height, nAPs, deadline, filesize, stillprob=0.4):
        self.agents         = None
        self._width         = width
        self._height        = height
        self._numOfLocation = width * height
        self._numOfAps      = nAPs
        self._still_p       = stillprob
        self._deadine       = deadline
        self._total_filesize = filesize
        self._listOfAPs     = None
        self._listOfAPsRate = None
        #self.ap_rate_list   = None

        if g_cellular_rate < 0:
            self.cellular_rate = utility.random_truncated_norm(g_cellular_rate_mean, g_cellular_rate_std)
        else:
            self.cellular_rate =  g_cellular_rate

        if g_list_ap_rate[0] < 0:
            self._listOfAPsRate   =   [utility.random_truncated_norm(int(g_ap_rate_mean),int(g_ap_rate_std)) for ap in range(int(nAPs))]
        else:
            self._listOfAPsRate = g_list_ap_rate

        utility.log_info("Cellular rate:"+self.cellular_rate.__str__())
        utility.log_info(str(nAPs) + " APs rate :" + self._listOfAPsRate.__str__())

        self.updata_ap_distribution()

        self.cellular_price =   g_cellular_price
        self.ap_price       =   g_ap_price

    def set_agents(self,agents):

        self.agents = agents

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def _loc2xy(self,loc): #convert location (start from index 0) to (x,y)
        x = loc / self._width
        y = loc % self._height

        return x, y

    def _xy2loc(self,x,y):#convert (x,y) to location (start from index 0)

        return x * self._width + y

    def _encode_state(self,loc,remain_file_size): # encode (location, remain file size) to state

        return remain_file_size * self._numOfLocation + loc

    def _decode_state(self, state): # decode state to (location, remain file size)
        # location = state % num of locations
        # size     = state / num of locations
        return state % self._numOfLocation, state / self._numOfLocation

    def _next_location(self, loc):
        next_locations = self.neighbour_locs(loc)
        p_current   = self._still_p
        num_of_locations = len(next_locations)
        p_next      = ( 1-self._still_p ) / num_of_locations

        problist = list()
        for i in range(num_of_locations):
            problist.append(p_next)

        next_locations.append(loc)
        problist.append(self._still_p)

        return utility.random_pick(next_locations,problist)

    def _energy_efficiency(self, rate):
        return g_Energy_a*exp(g_Energy_b * rate)

    def run(self, trials=10, experiments=1):
        average_values  = np.zeros((experiments,len(self.agents)))
        finished_rate   = np.zeros_like(average_values)
        deadline = self.agents[0].deadline
        for e in range(experiments):
            values = np.zeros((trials, len(self.agents)))
            finished = np.zeros_like(values)
            for t in range(trials):
                self.reset()
                route = self.generate_route(deadline)
                for i, agent in enumerate(self.agents):
                    for stage in range(agent.deadline):
                        l = route[stage]
                        state = self.encode_state(l, agent.b)
                        agent.set_state(state)
                        agent.set_t(stage)
                        agent.set_is_ap_exist(self.is_ap_exist(l))
                        action = agent.choose()
                        data_speed = 0
                        if action == 1:
                            data_speed = self.cellular_speed()
                        elif action == 2:
                            data_speed = self.ap_speed()
                        agent.observe(self.data_price(action),data_speed)

                    if agent.b != 0:
                        agent.total_cost += self.penalty(agent.b)
                    else:
                        finished[t,i]   = 1
                    values[t,i] = agent.total_cost
            average_values[e]   =   values.sum(axis=0)/trials
            finished_rate[e]    =   finished.sum(axis=0)/trials

        return average_values, finished_rate

    def run_mutiflow(self, trials=10, experiments=1):
        average_values  = np.zeros((experiments,len(self.agents)))
        average_e_cost  = np.zeros_like(average_values)
        finished_rate   = np.zeros_like(average_values)
        deadline = self.agents[0].deadlineVec[self.agents[0].numOfFlow-1]
        for e in range(experiments):
            values = np.zeros((trials, len(self.agents)))
            e_costs = np.zeros_like(values)
            finished = np.zeros_like(values)
            for t in range(trials):
                self.reset()
                route = self.generate_route(deadline)
                for i, agent in enumerate(self.agents):
                    for stage in range(int(agent.deadlineVec[agent.numOfFlow-1])):
                        l = route[stage]
                        #print "stage:",stage
                        #print "len(route):",len(route)
                        #print agent.deadlineVec
                        state = 0#self.encode_state(l, agent.b)
                        agent.set_state(state)
                        agent.set_t(stage)
                        agent.set_location(l)
                        agent.set_is_ap_exist(self.is_ap_exist(l))
                        # [ cellular rate, wi-fi rate]
                        rate_alloc_c_w = agent.choose()
                        action = 0
                        if agent.numOfFlow == 1:
                            if rate_alloc_c_w[0] > 0 and rate_alloc_c_w[1] == 0:
                                action = 1
                            elif rate_alloc_c_w[0] == 0 and rate_alloc_c_w[1] > 0:
                                action = 2
                        elif agent.numOfFlow > 1:
                        #    print "rate_alloc_c_w.sum(axis=1)", rate_alloc_c_w.sum(axis=1)
                        #    print "rate_alloc_c_w.sum(axis=1)[0]",rate_alloc_c_w.sum(axis=1)[0]
                        #    print "rate_alloc_c_w.sum(axis=1)[1]", rate_alloc_c_w.sum(axis=1)[1]
                            if rate_alloc_c_w.sum(axis=1)[0] > 0 and rate_alloc_c_w.sum(axis=1)[1] == 0:
                                # Cellular network is chosen
                                action = 1
                            elif rate_alloc_c_w.sum(axis=1)[0] == 0 and rate_alloc_c_w.sum(axis=1)[1] > 0:
                                # WiFi is chosen
                                action = 2
                        #print "action:", action
                        data_sum = rate_alloc_c_w.sum()
                        agent.observe(self.data_price(action),rate_alloc_c_w,self._energy_efficiency(data_sum)) # input: date rate, date price, energy efficiency
                    #if agent.b != 0:
                    #    agent.total_cost += self.penalty(agent.b)
                    #else:
                    #    finished[t,i]   = 1
                    #if np.all(agent.bVec==0):
                    finished[t,i] = len (np.where( agent.bVec == 0 )[0])
                    #else:
                    agent.total_cost += self.penalty( rate_alloc_c_w.sum())

                    values[t,i]     = agent.total_cost
                    e_costs[t,i]    = agent.total_e_cost
            average_values[e]   =   values.sum(axis=0)/trials
            average_e_cost[e]   =   e_costs.sum(axis=0)/trials
            finished_rate[e]    =   finished.sum(axis=0)/ ( trials * agent.numOfFlow )
#            average_values[e]   =   values.sum(axis=0)/trials
#            finished_rate[e]    =   finished.sum(axis=0)/trials

        return average_values, average_e_cost, finished_rate

    def encode_state(self,loc, remain_file_size):

        return self._encode_state(loc,remain_file_size)

    def decode_state(self,state):
        return  self._decode_state(state)

    def set_ap_rate_list(self,ap_rate_list):

        self._listOfAPsRate = ap_rate_list
        self._numOfAps  = len(ap_rate_list)
        self.updata_ap_distribution()

    def is_ap_exist(self, loc):
        if loc in self._listOfAPs:
            return True
        else:
            return False

    def num_of_locations(self):
        return self._numOfLocation

    def num_of_aps(self):
        return self._numOfAps

    def deadline(self):
        return self._deadine

    def total_filesize(self):
        return self._total_filesize

    def cellular_speed(self):

        return self.cellular_rate

    def ap_speed(self,location): # return the Wi-Fi date rate for the location

        return  self._listOfAPsRate[self._listOfAPs.index(location)]
        #self._listOfAPsRate(self._listOfAPs.index(location))

    def data_price(self,action):
        # action =(0,1,2)=(idle, Cellular, AP)
        if action == 0:
            return 0
        elif action == 1:
            return self.cellular_price
        elif action == 2:
            return self.ap_price

    def neighbour_locs(self,loc):
        neighbours = list()
        x,y = self._loc2xy(loc)
        if x == 0:
            if y == 0:
                neighbours.append((x + 1) * self._width + y)
                neighbours.append(x * self._width + y + 1)
            elif y == ( self._width - 1 ):
                neighbours.append(x * self._width + y - 1)
                neighbours.append((x + 1) * self._width + y)
            else:
                neighbours.append(x * self._width + y + 1)
                neighbours.append(x * self._width + y - 1)
                neighbours.append((x + 1) * self._width + y)
        elif x == (self._height - 1):
            if y == 0:
                neighbours.append((x - 1) * self._width + y)
                neighbours.append(x * self._width + y + 1)
            elif y == ( self._width - 1 ):
                neighbours.append(x * self._width + y - 1)
                neighbours.append((x - 1) * self._width + y)
            else:
                neighbours.append(x * self._width + y + 1)
                neighbours.append(x * self._width + y - 1)
                neighbours.append((x - 1) * self._width + y)
        else:
            if y == 0:
                neighbours.append(x * self._width + y + 1)
                neighbours.append((x - 1) * self._width + y)
                neighbours.append((x + 1) * self._width + y)
            elif y == ( self._width - 1 ):
                neighbours.append(x * self._width + y - 1)
                neighbours.append((x - 1) * self._width + y)
                neighbours.append((x + 1) * self._width + y)
            else:
                neighbours.append(x * self._width + y + 1)
                neighbours.append(x * self._width + y - 1)
                neighbours.append((x - 1) * self._width + y)
                neighbours.append((x + 1) * self._width + y)

        return neighbours

    def updata_ap_distribution(self):

        self._listOfAPs     = random.sample(range(self._numOfLocation),self._numOfAps)

    def generate_transistion(self,nActions, nFileSize):
        nDim_A = nActions
        nDim_S = self._numOfLocation * nFileSize
        P = np.zeros((nDim_A, nDim_S, nDim_S))
        R = np.zeros((nDim_A, nDim_S, nDim_S))

        self.updata_ap_distribution()

        for a in range(nDim_A):  # Actions={Idle, Cellular, WiFi}={0,1,2}
            bInvalidAction = False
            for s in range(nDim_S):  # State= location * remaining_file_size
                # decodde state = (location, remaining_file_size):
                loc, remaining_file_size = self._decode_state(s)
                data = 0
                reward = 0
                if a == 1:  # cellular
                    data = self.cellular_speed()
                elif a == 2:  # wifi
                    data = self.ap_speed()
                reward = data * self.data_price(a)
                # next_locations = next_locs(loc)
                next_locations = self.neighbour_locs(loc)
                p_next = (1 - self._still_p) / len(next_locations)
                next_locations.append(loc)

                for loc_new in next_locations:
                    remaining_file_size_new = remaining_file_size - data
                    if remaining_file_size_new < 0:
                        s_new = self._encode_state(loc_new, 0)
                        if loc_new == loc:
                            P[a, s, s_new] = self._still_p
                            R[a, s, s_new] = remaining_file_size * self.data_price(a)
                        else:
                            P[a, s, s_new] = p_next
                            R[a, s, s_new] = remaining_file_size * self.data_price(a)
                    else:
                        s_new = self._encode_state(loc_new,remaining_file_size_new)
                        if loc_new == loc:
                            P[a, s, s_new] = self._still_p
                            R[a, s, s_new] = reward
                        else:
                            P[a, s, s_new] = p_next
                            R[a, s, s_new] = reward

                    if ( self.is_ap_exist(loc) == False ) and ( a == 2 ): #choose action AP in non AP area, invalid action
                        remaining_file_size_new = remaining_file_size - data
                        if remaining_file_size_new < 0:
                            s_new = self._encode_state(loc_new, 0)
                            if loc_new == loc:
                                P[a, s, s_new] = self._still_p
                                R[a, s, s_new] = BIGVALUE
                            else:
                                P[a, s, s_new] = p_next
                                R[a, s, s_new] = BIGVALUE
                        else:
                            s_new = self._encode_state(loc_new, remaining_file_size_new)
                            if loc_new == loc:
                                P[a, s, s_new] = self._still_p
                                R[a, s, s_new] = BIGVALUE
                            else:
                                P[a, s, s_new] = p_next
                                R[a, s, s_new] = BIGVALUE

        return P, R

    def penalty(self,remain_file_size):

        return 10*remain_file_size*remain_file_size

    def generate_final_v_r(self):
        dim = self._numOfLocation * self._total_filesize
        finalV = np.zeros((dim))
        finalR = np.zeros((dim))
        k = 10
        for s in range(dim):
            loc, remaining_file_size = self._decode_state(s)
            finalV[s] = 10 * remaining_file_size * remaining_file_size
            finalR[s] = finalV[s]
        return finalV, finalR

    def generate_route(self, deadline):
        route = list()
        l_current = random.sample(range(self._numOfLocation),1)[0]
        route.append(l_current)
        for i in range(int(deadline)-1):
            l_next = self._next_location(l_current)
            route.append(l_next)
            l_current = l_next
        #print "Generated Route (DEADLINE=",deadline," )", "is: ",route
        return route

    def test(self):
#        self.test_xy2loc()
#        self.test_loc2xy()
#        self.test_neighbour_locs()
#        self.test_encode_state()
#        self.test_decode_state()
#        self.text_next_location()
#        self.text_energy_efficiency()
        a = 0

    def test_loc2xy(self):  # test function _loc2xy()
        print "-------This is the function test _loc2xy()--------"
        for l in range(self._numOfLocation):
            print "location ", l, "->", self._loc2xy(l)

    def test_xy2loc(self):  # test function _loc2xy()
        print "-------This is the function test _xy2loc()--------"
        for x in range(self._height):
            for y in range(self._width):
                print "(", x, ",", y, ")", "->", "location ", self._xy2loc(x, y)

    def test_neighbour_locs(self):
        print "-------This is the function test test_neighbour_locs()--------"
        for i in range(self._numOfLocation):
            print "Neighbour locations of ", i, " :", self.neighbour_locs(i)

    def test_encode_state(self):
        print "-------This is the function test _encode_state()--------"
        FILE_SIZE = 10
        for l in range(self._numOfLocation):
            for b in range(FILE_SIZE):
                print "(", l, ",", b, ")", "->", "state", self._encode_state(l, b)

    def test_decode_state(self):
        print "-------This is the function test _decode_state()--------"
        FILE_SIZE = 10
        NUM_OF_STATE = FILE_SIZE * self._numOfLocation
        for s in range(NUM_OF_STATE):
            print "state", s, "->", self._decode_state(s)

    def text_next_location(self):
        numOfStatic = 0
        print "static probability:", STATIC_P
        l=0
        for i in range(10000):
            l_next = self._next_location(l)
            if l_next == l:
                numOfStatic +=1
        print "num of static:", numOfStatic

    def text_energy_efficiency(self):
        print "======================================================================="
        print "======================================================================="
        print "======================================================================="
        print "This is the test function for energy efficiency changes with throughput"
        e_efficiency = 0
        for i in range(50):
            e_efficiency = self._energy_efficiency(i)
            print "rate = ", i, "energy efficiency = ", e_efficiency
        print "======================================================================="
        print "======================================================================="
        print "======================================================================="
class Environment(object):
    #              width(y) ->
    #    (0,0)________________
    #         |___|___|___|___|
    # height  |___|___|___|___|
    #  (x)    |___|___|___|___|
    #         |___|___|___|___|

    def __init__(self, width, height, nAPs, deadline, filesize, stillprob=0.4):
        self.agents         = None
        self._width         = width
        self._height        = height
        self._numOfLocation = width * height
        self._numOfAps      = nAPs
        self._still_p       = stillprob
        self._deadine       = deadline
        self._total_filesize = filesize
        self._listOfAPs     = None

        self.cellular_rate  =   int(g_cellular_rate_mean)
        self.ap_rate        =   int(g_ap_rate_mean)

        self.cellular_price =   g_cellular_price
        self.ap_price       =   g_ap_price

    def set_agents(self,agents):

        self.agents = agents

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def _loc2xy(self,loc): #convert location (start from index 0) to (x,y)
        x = loc / self._width
        y = loc % self._height

        return x, y

    def _xy2loc(self,x,y):#convert (x,y) to location (start from index 0)

        return x * self._width + y

    def _encode_state(self,loc,remain_file_size): # encode (location, remain file size) to state

        return remain_file_size * self._numOfLocation + loc

    def _decode_state(self, state): # decode state to (location, remain file size)
        # location = state % num of locations
        # size     = state / num of locations
        return state % self._numOfLocation, state / self._numOfLocation

    def _next_location(self, loc):
        next_locations = self.neighbour_locs(loc)
        p_current   = self._still_p
        num_of_locations = len(next_locations)
        p_next      = ( 1-self._still_p ) / num_of_locations

        problist = list()
        for i in range(num_of_locations):
            problist.append(p_next)

        next_locations.append(loc)
        problist.append(self._still_p)

        return utility.random_pick(next_locations,problist)

    def run(self, trials=10, experiments=1):
        average_values  = np.zeros((experiments,len(self.agents)))
        finished_rate   = np.zeros_like(average_values)
        deadline = self.agents[0].deadline
        for e in range(experiments):
            values = np.zeros((trials, len(self.agents)))
            finished = np.zeros_like(values)
            for t in range(trials):
                self.reset()
                route = self.generate_route(deadline)
                for i, agent in enumerate(self.agents):
                    for stage in range(agent.deadline):
                        l = route[stage]
                        state = self.encode_state(l, agent.b)
                        agent.set_state(state)
                        agent.set_t(stage)
                        agent.set_is_ap_exist(self.is_ap_exist(l))
                        action = agent.choose()
                        data = 0
                        if action == 1:
                            data = self.cellular_speed()
                        elif action == 2:
                            data = self.ap_speed()
                        agent.observe(self.data_price(action),data)

                    if agent.b != 0:
                        agent.total_cost += self.penalty(agent.b)
                    else:
                        finished[t,i]   = 1
                    values[t,i] = agent.total_cost
            average_values[e]   =   values.sum(axis=0)/trials
            finished_rate[e]    =   finished.sum(axis=0)/trials

        return average_values, finished_rate

    def run_mutiflow(self, trials=10, experiments=1):
        average_values  = np.zeros((experiments,len(self.agents)))
        finished_rate   = np.zeros_like(average_values)
        deadline = self.agents[0].deadlineVec[self.agents[0].numOfFlow-1]
        for e in range(experiments):
            values = np.zeros((trials, len(self.agents)))
            finished = np.zeros_like(values)
            for t in range(trials):
                self.reset()
                route = self.generate_route(deadline)
                for i, agent in enumerate(self.agents):
                    for stage in range(int(agent.deadlineVec[agent.numOfFlow-1])):
                        l = route[stage]
                        state = 0#self.encode_state(l, agent.b)
                        agent.set_state(state)
                        agent.set_t(stage)
                        agent.set_location(l)
                        agent.set_is_ap_exist(self.is_ap_exist(l))
                        action = agent.choose()
                        a=0
                        if np.all(action[0]==0):
                            a = 2
                        else:
                            a = 1

                        agent.observe(self.data_price(a),action)
                    #if agent.b != 0:
                    #    agent.total_cost += self.penalty(agent.b)
                    #else:
                    #    finished[t,i]   = 1
                    #if np.all(agent.bVec==0):
                    finished[t,i] = len (np.where( agent.bVec == 0 )[0])
                    #else:
                    agent.total_cost += self.penalty( action.sum())

                    values[t,i] = agent.total_cost
            average_values[e]   =   values.sum(axis=0)/trials
            finished_rate[e]    =   finished.sum(axis=0)/ ( trials * agent.numOfFlow )
#            average_values[e]   =   values.sum(axis=0)/trials
#            finished_rate[e]    =   finished.sum(axis=0)/trials

        return average_values, finished_rate

    def encode_state(self,loc, remain_file_size):

        return self._encode_state(loc,remain_file_size)

    def decode_state(self,state):
        return  self._decode_state(state)

    def is_ap_exist(self, loc):
        if loc in self._listOfAPs:
            return True
        else:
            return False

    def num_of_locations(self):
        return self._numOfLocation

    def num_of_aps(self):
        return self._numOfAps

    def deadline(self):
        return self._deadine

    def total_filesize(self):
        return self._total_filesize

    def cellular_speed(self):

        return self.cellular_rate

    def ap_speed(self):

        return self.ap_rate

    def data_price(self,action):
        # action =(0,1,2)=(idle, Cellular, AP)
        if action == 0:
            return 0
        elif action == 1:
            return self.cellular_price
        elif action == 2:
            return self.ap_price

    def neighbour_locs(self,loc):
        neighbours = list()
        x,y = self._loc2xy(loc)
        if x == 0:
            if y == 0:
                neighbours.append((x + 1) * self._width + y)
                neighbours.append(x * self._width + y + 1)
            elif y == ( self._width - 1 ):
                neighbours.append(x * self._width + y - 1)
                neighbours.append((x + 1) * self._width + y)
            else:
                neighbours.append(x * self._width + y + 1)
                neighbours.append(x * self._width + y - 1)
                neighbours.append((x + 1) * self._width + y)
        elif x == (self._height - 1):
            if y == 0:
                neighbours.append((x - 1) * self._width + y)
                neighbours.append(x * self._width + y + 1)
            elif y == ( self._width - 1 ):
                neighbours.append(x * self._width + y - 1)
                neighbours.append((x - 1) * self._width + y)
            else:
                neighbours.append(x * self._width + y + 1)
                neighbours.append(x * self._width + y - 1)
                neighbours.append((x - 1) * self._width + y)
        else:
            if y == 0:
                neighbours.append(x * self._width + y + 1)
                neighbours.append((x - 1) * self._width + y)
                neighbours.append((x + 1) * self._width + y)
            elif y == ( self._width - 1 ):
                neighbours.append(x * self._width + y - 1)
                neighbours.append((x - 1) * self._width + y)
                neighbours.append((x + 1) * self._width + y)
            else:
                neighbours.append(x * self._width + y + 1)
                neighbours.append(x * self._width + y - 1)
                neighbours.append((x - 1) * self._width + y)
                neighbours.append((x + 1) * self._width + y)

        return neighbours

    def updata_ap_distribution(self):

        self._listOfAPs     = random.sample(range(self._numOfLocation),self._numOfAps)

    def generate_transistion(self,nActions, nFileSize):
        nDim_A = nActions
        nDim_S = self._numOfLocation * nFileSize
        P = np.zeros((nDim_A, nDim_S, nDim_S))
        R = np.zeros((nDim_A, nDim_S, nDim_S))

        self.updata_ap_distribution()

        for a in range(nDim_A):  # Actions={Idle, Cellular, WiFi}={0,1,2}
            bInvalidAction = False
            for s in range(nDim_S):  # State= location * remaining_file_size
                # decodde state = (location, remaining_file_size):
                loc, remaining_file_size = self._decode_state(s)
                data = 0
                reward = 0
                if a == 1:  # cellular
                    data = self.cellular_speed()
                elif a == 2:  # wifi
                    data = self.ap_speed()
                reward = data * self.data_price(a)
                # next_locations = next_locs(loc)
                next_locations = self.neighbour_locs(loc)
                p_next = (1 - self._still_p) / len(next_locations)
                next_locations.append(loc)

                for loc_new in next_locations:
                    remaining_file_size_new = remaining_file_size - data
                    if remaining_file_size_new < 0:
                        s_new = self._encode_state(loc_new, 0)
                        if loc_new == loc:
                            P[a, s, s_new] = self._still_p
                            R[a, s, s_new] = remaining_file_size * self.data_price(a)
                        else:
                            P[a, s, s_new] = p_next
                            R[a, s, s_new] = remaining_file_size * self.data_price(a)
                    else:
                        s_new = self._encode_state(loc_new,remaining_file_size_new)
                        if loc_new == loc:
                            P[a, s, s_new] = self._still_p
                            R[a, s, s_new] = reward
                        else:
                            P[a, s, s_new] = p_next
                            R[a, s, s_new] = reward

                    if ( self.is_ap_exist(loc) == False ) and ( a == 2 ): #choose action AP in non AP area, invalid action
                        remaining_file_size_new = remaining_file_size - data
                        if remaining_file_size_new < 0:
                            s_new = self._encode_state(loc_new, 0)
                            if loc_new == loc:
                                P[a, s, s_new] = self._still_p
                                R[a, s, s_new] = BIGVALUE
                            else:
                                P[a, s, s_new] = p_next
                                R[a, s, s_new] = BIGVALUE
                        else:
                            s_new = self._encode_state(loc_new, remaining_file_size_new)
                            if loc_new == loc:
                                P[a, s, s_new] = self._still_p
                                R[a, s, s_new] = BIGVALUE
                            else:
                                P[a, s, s_new] = p_next
                                R[a, s, s_new] = BIGVALUE

        return P, R

    def penalty(self,remain_file_size):

        return 10*remain_file_size*remain_file_size

    def generate_final_v_r(self):
        dim = self._numOfLocation * self._total_filesize
        finalV = np.zeros((dim))
        finalR = np.zeros((dim))
        k = 10
        for s in range(dim):
            loc, remaining_file_size = self._decode_state(s)
            finalV[s] = 10 * remaining_file_size * remaining_file_size
            finalR[s] = finalV[s]
        return finalV, finalR

    def generate_route(self, deadline):
        route = list()
        l_current = random.sample(range(self._numOfLocation),1)[0]
        route.append(l_current)
        for i in range(int(deadline)-1):
            l_next = self._next_location(l_current)
            route.append(l_next)
            l_current = l_next
        #print "Generated Route (DEADLINE=",deadline," )", "is: ",route
        return route

    def test(self):
#        self.test_xy2loc()
#        self.test_loc2xy()
#        self.test_neighbour_locs()
#        self.test_encode_state()
#        self.test_decode_state()
        self.text_next_location()

    def test_loc2xy(self):  # test function _loc2xy()
        print "-------This is the function test _loc2xy()--------"
        for l in range(self._numOfLocation):
            print "location ", l, "->", self._loc2xy(l)

    def test_xy2loc(self):  # test function _loc2xy()
        print "-------This is the function test _xy2loc()--------"
        for x in range(self._height):
            for y in range(self._width):
                print "(", x, ",", y, ")", "->", "location ", self._xy2loc(x, y)

    def test_neighbour_locs(self):
        print "-------This is the function test test_neighbour_locs()--------"
        for i in range(self._numOfLocation):
            print "Neighbour locations of ", i, " :", self.neighbour_locs(i)

    def test_encode_state(self):
        print "-------This is the function test _encode_state()--------"
        FILE_SIZE = 10
        for l in range(self._numOfLocation):
            for b in range(FILE_SIZE):
                print "(", l, ",", b, ")", "->", "state", self._encode_state(l, b)

    def test_decode_state(self):
        print "-------This is the function test _decode_state()--------"
        FILE_SIZE = 10
        NUM_OF_STATE = FILE_SIZE * self._numOfLocation
        for s in range(NUM_OF_STATE):
            print "state", s, "->", self._decode_state(s)

    def text_next_location(self):
        numOfStatic = 0
        print "static probability:", STATIC_P
        l=0
        for i in range(10000):
            l_next = self._next_location(l)
            if l_next == l:
                numOfStatic +=1
        print "num of static:", numOfStatic

class Agent(object):

    def __init__(self, policy, route=0):
        self.policy         =   policy
        self.t              =   0
        self.costvector     =   list()

        self.location       =   0
        self.route          =   route

        self.actionlist     =  list()

        self.total_cost     =   0
        self.total_e_cost   =   0
        self.state          =   0
        self.isAPExist      =   False

        self.theta          =   g_Theta

    def reset(self):
        self.total_cost     = 0
        self.total_e_cost     = 0
        self.isAPExist      = False
        self.policy.reset()

    def choose(self):

        return self.policy.choose(self)

    def set_state(self,state):
        self.state = state

    def set_is_ap_exist(self,is_ap_exist):
        self.isAPExist = is_ap_exist

    def set_t(self,t):
        self.t = t

    def set_location(self,loc):
        self.location = loc

    def set_theta(self,newtheta):
        self.theta = newtheta

    def observe(self, data_price, action, energy_efficiency):

        return 0


class SingleflowAgent(Agent):

    def __init__(self, policy, deadline, filesize, route=0):
        super(SingleflowAgent, self).__init__(policy,route)
        self.deadline   =   deadline
        self.filesize   =   filesize
        self.b          =   None

    def reset(self):
        super(SingleflowAgent, self).reset()
        self.b              = self.filesize - 1

    def observe(self, data_price, action, energy_efficiency):

        #self.costvector.append(cost)
        if self.b < data:
            self.total_cost += self.b * dataprice
        else:
            self.total_cost += data * dataprice

        self.b = utility.max(0,self.b - data)

class MutiflowAgent(Agent):

    def __init__(self, policy, deadlineVec, filesizeVec, route=0):
        super(MutiflowAgent, self).__init__(policy,route)

        assert len(deadlineVec) == len(filesizeVec), "length of deadline vector and filesize vector should be the same."
        self.deadlineVec    =   deadlineVec
        self.filesizeVec    =   filesizeVec
        self.numOfFlow      =   len(deadlineVec)
        self.numOfActiveFlow = len(deadlineVec)
        self.bVec           =   np.array(filesizeVec)
        self.listActions    = None

    def reset(self):
        super(MutiflowAgent, self).reset()
        self.bVec              = np.array(self.filesizeVec) - 1

    def set_deadline_list(self,deadline_list):
        self.deadlineVec = deadline_list
        self.numOfFlow  = len(deadline_list)
        self.numOfActiveFlow = self.numOfActiveFlow

    def set_filesize_list(self,filesize_list):
        self.filesizeVec = filesize_list
        self.numOfFlow  = len(filesize_list)
        self.numOfActiveFlow = self.numOfActiveFlow
        self.bVec           =   np.array(filesize_list)

    def observe(self, data_price, action, energy_efficiency):
#    def observe(self, dataprice, action):
        action_sum = action.sum(axis=0)
        for i in range(self.numOfFlow):
            if self.bVec[i] < action_sum[i]:
                self.total_cost += self.bVec[i] * data_price
#                self.total_e_cost += self.theta * energy_efficiency * self.bVec[i]
                self.total_e_cost += energy_efficiency * self.bVec[i]
            else:
                self.total_cost += action_sum[i] * data_price
#                self.total_e_cost += self.theta * energy_efficiency * action_sum[i]
                self.total_e_cost += energy_efficiency * action_sum[i]
            self.bVec[i] = utility.max(0, self.bVec[i] - action_sum[i])

        self.numOfActiveFlow = np.count_nonzero(self.bVec)

class Policy(object):

    def __str__(self):
        return 'generic policy'

    def reset(self):
        return 0

    def choose(self,agent):
        return 0

class MdpPolicy(Policy):

    def __init__(self,env):

        self.env = env

#        finalV, finalR = env.generate_final_v_r()
#        P, R = env.generate_transistion(3, int(g_Total_Size))

#        self.mdp = mdptoolbox.mdp.FiniteHorizon(P, R, 0.99999999999, env.deadline(), finalV)
#        self.mdp.run()

    def __str__(self):
        return 'policy based on Markov Decision Process (MDP)'

    def reset(self):

        #finalV, finalR = self.env.generate_final_v_r()
        #P, R = self.env.generate_transistion(3, int(g_Total_Size))
        self.env.update_ap_distribution()
        #self.mdp = mdptoolbox.mdp.FiniteHorizon(P, R, 0.999999999, self.env.deadline(), finalV)
        #self.mdp.run()

    def choose(self,agent):

        return self.mdp.policy[agent.state, agent.t]

class OnTheSpotPolicy(Policy):

    def __str__(self):
        return 'policy based on the spot offloading (OTSO)'

    def choose(self,agent):
        action = 1
        if agent.isAPExist is True:
            action = 2

        return action

class NoOffloadingPolicy(Policy):

    def __str__(self):
        return 'no offloading policy'

    def choose(self,agent):
        #action = 1 #always choose cellular
        return 1

class MutiflowOTSOPolicy(Policy):

    def __init__(self,env):

        self.env                    = env
        self.deadline_remain_list   = None
        self.weight_list            = None
        self.a_cellular             = None
        self.a_ap                   = None
        self.T_th                   = 4

    def __str__(self):
        return 'On the spot offloading policy for multi-flow'

    def reset(self):

        self.env.updata_ap_distribution()

    def choose(self,agent):
        #self.deadline_remain_list   = list()  #np.zeros((agent.numOfFlow))
        self.weight_list            = list()   #np.zeros((agent.numOfFlow))
        k = max(agent.deadlineVec)
        for deadline in agent.deadlineVec:
            weight = 0
            if deadline > agent.t:
                #left_t = deadline - agent.t
                #self.deadline_remain_list.append(left_t)
                weight = 1
            #else:
                #self.deadline_remain_list.append(0)
            self.weight_list.append(weight)

        self.weight_filesize_list = np.array(agent.bVec)
        self.weight_filesize_list[np.where(self.weight_filesize_list>0)]=1
        self.weight_list = np.multiply(self.weight_list,self.weight_filesize_list)
        self.weight_list = utility.normalize(self.weight_list)

        self.a_cellular = np.zeros((agent.numOfFlow))
        self.a_ap       = np.zeros((agent.numOfFlow))

        if agent.isAPExist == True:
            self.a_ap = np.multiply(self.weight_list, self.env.ap_speed(agent.location))
        else:
            self.a_cellular = np.multiply(self.weight_list, self.env.cellular_speed())

        return np.concatenate((self.a_cellular,self.a_ap)).reshape(2,agent.numOfFlow)

class MutiflowHeuristicPolicy(Policy):

    def __init__(self,env):

        self.env                    = env
        self.deadline_remain_list   = None
        self.weight_list            = None
        self.a_cellular             = None
        self.a_ap                   = None
        self.T_th                   = 4
        #self.r_th                   = g_Theta
    def __str__(self):
        return 'Heuristic policy for multi-flow'

    def reset(self):
        return 0
#        finalV, finalR = self.env.generate_final_v_r()
#        P, R = self.env.generate_transistion(3, int(g_Total_Size))
#
#        self.mdp = mdptoolbox.mdp.FiniteHorizon(P, R, 0.999999999, self.env.deadline(), finalV)
#        self.mdp.run()
        self.env.updata_ap_distribution()

    def choose(self,agent):
        self.deadline_remain_list   = list()  #np.zeros((agent.numOfFlow))
        self.weight_list            = list()   #np.zeros((agent.numOfFlow))
        k = max(agent.deadlineVec)
        for deadline in agent.deadlineVec:
            weight = 0
            if deadline > agent.t:
                left_t = deadline - agent.t
                self.deadline_remain_list.append(left_t)
                weight = k/float(left_t)
            else:
                self.deadline_remain_list.append(0)
            self.weight_list.append(weight)

        self.weight_list = utility.normalize(np.array(self.weight_list))
        self.weight_filesize_list = utility.normalize(agent.bVec)
        self.weight_list = np.multiply(self.weight_list,self.weight_filesize_list)#
        self.weight_list = utility.normalize(self.weight_list)

        self.a_cellular = np.zeros((agent.numOfFlow))
        self.a_ap       = np.zeros((agent.numOfFlow))

        self.T_th = min(agent.filesizeVec)*agent.numOfActiveFlow / self.env.cellular_speed() + 1
        #print "Time threshold:", self.T_th, "Rate threshoud:", agent.theta
        if agent.isAPExist == True and self.env.ap_speed(agent.location) > agent.theta:
            self.a_ap = np.multiply(self.weight_list, self.env.ap_speed(agent.location))
        else:
            if min(self.deadline_remain_list) <= self.T_th:
                self.a_cellular = np.multiply(self.weight_list, self.env.cellular_speed())

        return np.concatenate((self.a_cellular,self.a_ap)).reshape(2,agent.numOfFlow)

def execute():

    env = Environment(int(g_Location_Width),int(g_Location_Height),\
                      int(g_Number_APs),\
                      int(g_Deadline_T),\
                      int(g_Total_Size),\
                      g_Still_Probability)

    mdp_policy  = MdpPolicy(env)
    otso_policy = OnTheSpotPolicy()
    no_policy   = NoOffloadingPolicy()

    agents = [SingleflowAgent(mdp_policy,int(g_Deadline_T), int(g_Total_Size)),
              SingleflowAgent(otso_policy,int(g_Deadline_T), int(g_Total_Size)),
              SingleflowAgent(no_policy, int(g_Deadline_T), int(g_Total_Size))]

    env.set_agents(agents)

    return env.run(100,1)

def execute_multiflow():

    env = EnvironmentEX(int(g_Location_Width),int(g_Location_Height),\
                      int(g_Number_APs),\
                      int(g_Deadline_T),\
                      int(g_Total_Size),\
                      g_Still_Probability)

    #print env
    heuristic_policy  = MutiflowHeuristicPolicy(env)
    otso_policy = MutiflowOTSOPolicy(env)
#    no_policy   = NoOffloadingPolicy()

#    agents = [MutiflowAgent(heuristic_policy,g_list_T, g_list_B)]
    #agents = [MutiflowAgent(otso_policy,g_list_T, g_list_B)]
    agents = [MutiflowAgent(heuristic_policy,g_list_T, g_list_B),
              MutiflowAgent(otso_policy,g_list_T, g_list_B)]
 #             Agent(no_policy, int(g_Deadline_T), int(g_Total_Size))]

    env.set_agents(agents)

    print env.run_mutiflow(1,1)
    return

def energy_consumption_diff_preference():
    e_DP            = 0
    e_Heuristic     = 0
    e_DAWN          = 0
    e_OnTheSpot     = 0
    e_NoOffloaing   = 0

    utility.create_output_file("energywithpreference.csv")
    env = EnvironmentEX(int(g_Location_Width),int(g_Location_Height),\
                      int(g_Number_APs),\
                      int(g_Deadline_T),\
                      int(g_Total_Size),\
                      g_Still_Probability)

    #print env
    heuristic_policy  = MutiflowHeuristicPolicy(env)
#    otso_policy = MutiflowOTSOPolicy(env)
#    no_policy   = NoOffloadingPolicy()

    list_theta = [0,1,2,3,4]
    agents = []
    for theta in list_theta:
        agent = MutiflowAgent(heuristic_policy, g_list_T, g_list_B)
        agent.set_theta(theta)
        agents.append(agent)

    #agents = [MutiflowAgent(otso_policy,g_list_T, g_list_B)]
#    agents = [MutiflowAgent(heuristic_policy,g_list_T, g_list_B),
#              MutiflowAgent(otso_policy,g_list_T, g_list_B)]
 #             Agent(no_policy, int(g_Deadline_T), int(g_Total_Size))]

    env.set_agents(agents)

    env.test()
    results =  env.run_mutiflow(100,1)
    print results
 #   print [str(theta)+"," for theta in results[0][0]]
   # utility.log_info(",".join(str(x) for x in probMatrix.sum(1)))
    utility.save_to_output(",".join(str(theta) for theta in list_theta))
    utility.save_to_output(",".join(str(cost) for cost in results[0][0]))
    utility.save_to_output(",".join(str(energy) for energy in results[1][0]))
    utility.save_to_output(",".join(str(finishrate) for finishrate in results[2][0]))
 #   utility.save_to_output(str(cost)+"," for cost in results[0])
 #   utility.save_to_output(str(energy)+"," for energy in results[1])
 #   utility.save_to_output(str(finishrate)+"," for finishrate in results[2])

    utility.close_output_file()

    return

def cost_energy_diff_flows():
    e_DP            = 0
    e_Heuristic     = 0
    e_DAWN          = 0
    e_OnTheSpot     = 0
    e_NoOffloaing   = 0

    utility.create_output_file("cost_energy_flows.csv")
    env = EnvironmentEX(int(g_Location_Width),int(g_Location_Height),\
                      int(g_Number_APs),\
                      int(g_Deadline_T),\
                      int(g_Total_Size),\
                      g_Still_Probability)

    #print env
    heuristic_policy  = MutiflowHeuristicPolicy(env)
    otso_policy = MutiflowOTSOPolicy(env)

    g_list_B2 = [500,550]
    g_list_B3 = [500,550,600]
    g_list_B4 = [500,550,600,650]

    g_list_T2 = [140,280]
    g_list_T3 = [140,280,420]
    g_list_T4 = [140,280,420,560]

    agent1 = MutiflowAgent(heuristic_policy,g_list_T, g_list_B)

    agent2 = MutiflowAgent(heuristic_policy,g_list_T2, g_list_B2)
    agent3 = MutiflowAgent(heuristic_policy,g_list_T3, g_list_B3)
    agent4 = MutiflowAgent(heuristic_policy,g_list_T4, g_list_B4)


    agent11 = MutiflowAgent(otso_policy,g_list_T, g_list_B)
    agent12 = MutiflowAgent(otso_policy,g_list_T2, g_list_B2)
    agent13 = MutiflowAgent(otso_policy,g_list_T3, g_list_B3)
    agent14 = MutiflowAgent(otso_policy,g_list_T4, g_list_B4)

    agent_list = [agent1,agent2,agent3,agent4,agent11,agent12,agent13,agent14]
    cost_list   = []
    energy_list = []
    for agent in agent_list:
        agents = []
        agents.append(agent)
        env.set_agents(agents)
        results = env.run_mutiflow(100,1)
        print results
        cost_list.append(results[0][0][0])
        energy_list.append(results[1][0][0])

    utility.save_to_output(",".join(str(cost) for cost in cost_list))
    utility.save_to_output(",".join(str(energy) for energy in energy_list))

    utility.close_output_file()

    return

def cost_energy_diff_aps():

    utility.create_output_file("cost_energy_aps.csv")

    rate_list_aps0 = [0]
    rate_list_aps2 = [2,18]
    rate_list_aps4 = [1,2,15,20]
    rate_list_aps6 = [1,2,4,9,15,22]
    rate_list_aps8 = [1,2,3,4,9,15,20,22]
    rate_list_aps10 = [1,2,3,4,9,15,18,20,22,23]

    ap_rate_list = [rate_list_aps0,rate_list_aps2,rate_list_aps4,rate_list_aps6,rate_list_aps8,rate_list_aps10]

    cost_h_list   = []
    cost_b_list   = []
    energy_h_list = []
    energy_b_list = []
    for ap_rates in ap_rate_list:
        env = EnvironmentEX(int(g_Location_Width), int(g_Location_Height), \
                            int(g_Number_APs), \
                            int(g_Deadline_T), \
                            int(g_Total_Size), \
                            g_Still_Probability)

        env.set_ap_rate_list(ap_rates)

        heuristic_policy = MutiflowHeuristicPolicy(env)
        otso_policy = MutiflowOTSOPolicy(env)

        agents = [MutiflowAgent(heuristic_policy,g_list_T, g_list_B),
                  MutiflowAgent(otso_policy,g_list_T, g_list_B)]

        env.set_agents(agents)
        results = env.run_mutiflow(100,1)
        print results
       # print results[0]
       # print results[0][0]
        print results[0][0][0]
        print results[0][0][1]
        print results[1][0][0]
        print results[1][0][1]
        cost_h_list.append(results[0][0][0])
        cost_b_list.append(results[0][0][1])
        energy_h_list.append(results[1][0][0])
        energy_b_list.append(results[1][0][1])

    utility.save_to_output(",".join(str(cost) for cost in cost_h_list))
    utility.save_to_output(",".join(str(cost) for cost in cost_b_list))
    utility.save_to_output(",".join(str(energy) for energy in energy_h_list))
    utility.save_to_output(",".join(str(energy) for energy in energy_b_list))
    utility.close_output_file()

    return

def finish_rate_diff_aps():

    utility.create_output_file("finishrate_aps.csv")

    rate_list_aps0 = [0]
    rate_list_aps2 = [2,14]
    rate_list_aps4 = [1,2,14,18]
    rate_list_aps6 = [1,2,14,16,18,22]
    rate_list_aps8 = [1,2,14,16,18,19,20,24]
    rate_list_aps10 = [1,2,14,15,16,17,18,20,22,21]

    ap_rate_list = [rate_list_aps0,rate_list_aps2,rate_list_aps4,rate_list_aps6,rate_list_aps8,rate_list_aps10]

    finish_rate_h_list   = []
    finish_rate_b_list   = []
    for ap_rates in ap_rate_list:
        env = EnvironmentEX(int(g_Location_Width), int(g_Location_Height), \
                            int(g_Number_APs), \
                            int(g_Deadline_T), \
                            int(g_Total_Size), \
                            g_Still_Probability)

        env.set_ap_rate_list(ap_rates)

        heuristic_policy = MutiflowHeuristicPolicy(env)
        otso_policy = MutiflowOTSOPolicy(env)

        agents = [MutiflowAgent(heuristic_policy,g_list_T, g_list_B),
                  MutiflowAgent(otso_policy,g_list_T, g_list_B)]

        env.set_agents(agents)
        results = env.run_mutiflow(100,1)
        print results
       # print results[0]
       # print results[0][0]
       # print results[0][0][0]
       # print results[0][0][1]
       # print results[2][0][0]
       # print results[2][0][1]
        #finish_rate_h_list = []
        finish_rate_h_list.append(results[2][0][0])
        finish_rate_b_list.append(results[2][0][1])

    utility.save_to_output(",".join(str(rate) for rate in finish_rate_h_list))
    utility.save_to_output(",".join(str(rate) for rate in finish_rate_b_list))
    utility.close_output_file()

    return

def execute_diff_deadline(deadline_max):
    global g_Deadline_T

    for deadline in range(3,deadline_max):
        g_Deadline_T = deadline
        print execute()

def main():
    print '\n============================================================='
    print '\n\tThis is Simulation Program for Mobile Data Offloading!'
    print '\n\tAuthor: Cheng ZHANG'
    print '\n\t  Date: 2017/2/9'
    print '\n============================================================='
    utility.init_log_file("log.log")
    utility.log_info("Enter function: "+main.__name__)
##    utility.create_output_file()
##
    if ( params_def_init() is False ):
        utility.log_error("Fail to initialize parameters, abort the program")
        print "[ERROR] Fail to initialize parameters, abort the program, see log.log file for detail information."
        utility.close_log_file()
        return

    deadline_max = 18
#    execute_diff_deadline(deadline_max)
   # print extractlist("g_T_")
   # print extractlist("g_B_")
 #   print g_list_B
 #   print g_list_T
#    energy_consumption_diff_preference()
#    cost_diff_flows()
#    execute_multiflow()
#    cost_energy_diff_aps()
    finish_rate_diff_aps()
#    i = 0
#    i+=1
    # ql = mdptoolbox.mdp.QLearning(P, R, 0.96)
    # ql.run()
    # ql.Q
    # ql.policy
if __name__ == '__main__':
    main()