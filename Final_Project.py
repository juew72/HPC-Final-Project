from collections import namedtuple
import sys
import math
import time
import copy
import matplotlib.pyplot as plt
import scipy
import numpy as np
import random
from tqdm import tqdm
from matplotlib import animation as animation

fig = plt.figure()
generations = 600

'''
Create a new traffic simulation object for three-lane model. Cars are distributed randomly along the road, among the three lanes and start with random velocities.
Inputs:
 length (int) The number of cells in one lane.
 density (float) The fraction of cells that have a car on them.
 vmax (int) The maximum speed in car cells per update step.
 slow (float) The probability that a car will randomly slow down by 1 during an update step.
 change (float) The probability that a car will switch to the other lane.
'''

class TrafficSimulator(object):
    def __init__(self,density, change):
        self.length = 100
        self.density = density
        self.vmax = 5
        self.slow = 0.5 #prop of slowing down
        self.change = change  #prop of changing lane
        #A state is represented as a 3 dimensional array: 
        self.current_state = np.zeros((3,self.length)) 
        self.next_state = np.zeros((3,self.length))
        # Track the time steps and total number of cars that passed the simulation
        # boundary to estimate average traffic flow.
        self.cumulative_traffic_flow = 0
        self.time_step = 0
        
    def initialize(self):
        random_indices = np.random.randint(
            0,self.length*3-1,
            size=int(round(self.density*3*self.length)))
        pos_indices=random_indices%self.length
        lane_indices=random_indices//self.length
        self.current_state.fill(-1)
        for i in range(len(random_indices)):
            self.current_state[lane_indices[i]][pos_indices[i]]= scipy.random.randint(low = 0, 
                                                                  high = self.vmax +1)
        
    def display(self):     
        print(''.join('.' if int(x) == -1 else str(int(x)) for x in self.current_state[0]))
        print(''.join('.' if int(x) == -1 else str(int(x)) for x in self.current_state[1]))
        print(''.join('.' if int(x) == -1 else str(int(x)) for x in self.current_state[2]))
        
    def calculate_gaps(self,this_lane,other_lane):
        gaps=[]
        for i in range(0,self.length):
            gap=0
            gap0=0
            gapb=0
            if self.current_state[this_lane][i] != -1:
                #if there is another car right at the space parallel to the current car 
                #the gap0 and gapb are assigned -1, which means car can't switch lane.
                if self.current_state[other_lane][i] != -1:
                    gap0 = -1
                    gapb = -1
                j = i
                j0 = i
                jb = i
                found_next=False
                found_next0=False
                found_back=False
                #Measure gap from this car to next car on the same lane:
                #if there is only one car on this lane, assign gap to be 100. 
                count_this_lane=0
                for car in self.current_state[this_lane]:
                    if car != -1:
                        count_this_lane+=1
                if count_this_lane == 1:
                    gap =100 
                else:
                    while not found_next:
                        j+=1
                        if self.current_state[this_lane][j%len(self.current_state[this_lane])] == -1:
                            gap+=1
                        else:
                            found_next=True
               
                #if there is no car the other lane, assign gap0 and gapb to 100. 
                count_other_lane=0
                for c in self.current_state[other_lane]:
                    if c != -1:
                        count_other_lane+=1
                if count_other_lane == 0:
                    gap0=100
                    gapb=100
                #Else: 
                else:
                    #Measure gap from this car to next car on the other lane:
                    while not found_next0:
                        j0+=1
                        if self.current_state[other_lane][j0%len(self.current_state[other_lane])] == -1:
                            gap0+=1
                        else:
                            found_next0=True
                    #Measure gap from this car to the previous car on the other lane:
                    while not found_back:
                        jb-=1
                        if self.current_state[other_lane][jb%len(self.current_state[other_lane])] == -1:
                            gapb+=1
                        else:
                            found_back=True
            gaps.append((gap,gap0,gapb))      
        return (gaps)
            
    def change_lane(self):
        #Get the gaps for all cars on lane 0 in relation to lane 1
        gap01 = self.calculate_gaps(0,1)
        #Get the gaps for all cars on lane 2 in relation to lane 1
        gap21 = self.calculate_gaps(2,1)
        #Get the gaps for all cars on lane 1 in relation to lane 0
        gap10 = self.calculate_gaps(1,0)
        #Get the gaps for all cars on lane 1 in relation to lane 2
        gap12 = self.calculate_gaps(1,2)
        
        #Make a copy of each of the lane: 
        temp0 = self.current_state[0].copy()
        temp1 = self.current_state[1].copy()
        temp2 = self.current_state[2].copy()
        #Cars switch from lane 1 to lane 0 or lane 2
        for j in range(0,self.length):
            cond10_1 = self.current_state[1][j]+1 > gap10[j][0]
            cond10_2 = self.current_state[1][j]+1 < gap10[j][1]
            cond10_3 = self.vmax < gap10[j][2]
            cond10_4 = scipy.random.random() < self.change
            cond12_1 = self.current_state[1][j]+1 > gap12[j][0]
            cond12_2 = self.current_state[1][j]+1 < gap12[j][1]
            cond12_3 = self.vmax < gap12[j][2]
            cond12_4 = scipy.random.random() < self.change
            #If a car can switch to both lane 0 and lane 2, toss a fair coin.
            if cond10_1 and cond10_2 and cond10_3 and cond10_4 \
            and cond12_1 and cond12_2 and cond12_3 and cond12_4:
                if scipy.random.random() < 0.5:
                    temp0[j] = self.current_state[1][j]
                else:
                    temp2[j] = self.current_state[1][j]
                temp1[j] = -1
            #If a car can only switch to lane 0:
            elif cond10_1 and cond10_2 and cond10_3 and cond10_4:
                temp0[j] = self.current_state[1][j]
                temp1[j] = -1
            #If a car can only switch to lane 2: 
            elif cond12_1 and cond12_2 and cond12_3 and cond12_4:
                temp2[j] = self.current_state[1][j]
                temp1[j] = -1 
            else:
                continue
        #Cars switch from lane 0 to lane 1 or from lane 2 to lane 1: 
        for i in range(0,self.length):
            cond01_1 = self.current_state[0][i]+1 > gap01[i][0]
            cond01_2 = self.current_state[0][i]+1 < gap01[i][1]
            cond01_3 = self.vmax < gap01[i][2]
            cond01_4 = scipy.random.random() < self.change
            cond21_1 = self.current_state[0][i]+1 > gap21[i][0]
            cond21_2 = self.current_state[0][i]+1 < gap21[i][1]
            cond21_3 = self.vmax < gap21[i][2]
            cond21_4 = scipy.random.random() < self.change
            #If a space in lane 1 can be switched with both lane 0 and lane 2, we toss a coin:
            if cond01_1 and cond01_2 and cond01_3 and cond01_4\
            and cond21_1 and cond21_2 and cond21_3 and cond21_4:
                if scipy.random.random() < 0.5:
                    temp1[i] = self.current_state[0][i]
                    temp0[i] = -1
                else:
                    temp1[i] = self.current_state[2][i]
                    temp2[i] = -1 
            #If only lane 0 can switch to lane 1:
            elif cond01_1 and cond01_2 and cond01_3 and cond01_4:
                temp1[i] = self.current_state[0][i]
                temp0[i] = -1
            #If only lane 2 can switch to lane 1: 
            elif cond21_1 and cond21_2 and cond21_3 and cond21_4:
                temp1[i] = self.current_state[2][i]
                temp2[i] = -1
            else:
                continue
        #Update all the three lanes simultaneously: 
        self.current_state[0] = temp0.copy()
        self.current_state[1] = temp1.copy()
        self.current_state[2] = temp2.copy()
                
    def step_for_one_lane(self,lane):
        for i in range(0, self.length):
            if self.current_state[lane][i] != -1:
                #Calculate distance from this car to next car on the same lane: 
                distance_to_next=0
                j=i
                found_next=False
                while not found_next:
                    j+=1
                    if self.current_state[lane][j%len(self.current_state[lane])] == -1:
                        distance_to_next+=1
                    else:
                        found_next=True
                #Rule 1: Acceleration
                if self.current_state[lane][i] < self.vmax and distance_to_next > self.current_state[lane][i]:
                    self.current_state[lane][i] = min(self.current_state[lane][i] + 1, self.vmax)
                #Rule 2: Deterministic Deceleration
                if distance_to_next < self.current_state[lane][i]: 
                    self.current_state[lane][i] = distance_to_next
                #Rule 3: Random Deceleration
                if scipy.random.random() < self.slow:
                    self.current_state[lane][i] = max(self.current_state[lane][i] - 1,0)
        #Rule 4:  Move cars forward using their updated velocities
        self.next_state[lane].fill(-1)
        for i in range(0, self.length):
            v = self.current_state[lane][i]
            if v != -1: 
                self.next_state[lane][int((i+v)%self.length)] = v
    def after_move_car(self):
        #Move cars in three lanes simultaneously
        self.step_for_one_lane(0)
        self.step_for_one_lane(1)
        self.step_for_one_lane(2)
        # Swap next state and current state
        self.current_state, self.next_state = self.next_state, self.current_state
        self.time_step += 1
        for i in range(self.vmax):
            if self.current_state[0][i] > i:
                self.cumulative_traffic_flow += 1
            if self.current_state[1][i] > i:
                self.cumulative_traffic_flow += 1
            if self.current_state[2][i] > i:
                self.cumulative_traffic_flow += 1

def display_states(dens):
    test = TrafficSimulator(density=dens, change=1)
    test.initialize()
    print ("Initial State: ")
    test.display()
    for i in range(0,generations):
        print("Step %s" % (i+1))
        test.change_lane()
        test.after_move_car()
        test.display()

display_states(0.3)
