import copy

import numpy as np
from numpy.linalg import inv
import math
import matplotlib.pyplot as plt

class incrementalOffPolicy(object):
	def __init__(self, filename):
		actualMap = np.array(self.txt_to_obs_map('track1.txt'))
		reviseMap = np.empty(actualMap.shape)
		for i in range(0,len(actualMap)):
			reviseMap[i,] = actualMap[len(actualMap)-1-i,]
		self.map = copy.copy(np.transpose(reviseMap))
		self.start = np.empty([1,2])
		for i in range(0, self.map.shape[0]):
			if self.map[i,0] == 0:
				self.start = np.vstack((self.start,(i,0)))
		self.start = np.delete(self.start,(0), axis=0)
		self.finish = np.empty([1,2])
		for i in range(0, self.map.shape[1]):
			if self.map[self.map.shape[0]-1,i] == 0:
				self.finish = np.vstack((self.finish,(self.map.shape[0]-1,i)))
		self.finish = np.delete(self.finish, (0), axis=0)
		self.initialVelocity = [0,0]
		self.PossibleActions = np.array(((-1,-1), (-1,0), (-1,1), 
										(0,-1), (0,0), (0,1),
										(1,-1), (1,0), (1,1)))
		try:
			self.Q = np.loadtxt('q.txt')
		except:
			self.Q = np.empty(shape=(self.map.size*6*6,9))

		self.C = np.zeros(shape=(self.map.size*6*6,9))
		self.finishState = np.empty(shape=(1,4))

	def txt_to_obs_map(self, file_name):
	    with open(file_name) as inputFile:
	        return [[int(i) for i in line.strip().split('\t')] for line in inputFile]

	def behaviourPolicy(self):
		return np.random.randint(9)

	def targetPolicy(self, previousState):
		if(previousState[1] ==0):
			xVel = 0
			yVel = 1
		elif (previousState[1] < 28):
			if(previousState[3] == 0):
				xVel = 0
				yVel = 1
			else:
				xVel = 0
				yVel = 0
		elif(previousState[1] >= 28):
			if (previousState[2] == 0):
				xVel = 1
			else:
				xVel = 0
			if(previousState[3] == 0):
				yVel = 0
			else:
				yVel = -1
		return np.array((xVel, yVel))

	def generateEpisode(self):
		# print(self.start, self.start.shape)
		previousState = np.append(copy.copy(
							self.start[np.random.randint(self.start.shape[0])]), [0,0])
		episode = np.zeros(shape=(1,7))
		while True:
			actionIndex = self.behaviourPolicy()
			action = copy.copy(self.PossibleActions[actionIndex])
			velocity_x = max(min((previousState[2]+action[0]),5),0)
			velocity_y = max(min((previousState[3]+action[1]),5),0)
			nextState = np.array(((previousState[0] + velocity_x),
									(previousState[1] + velocity_y), velocity_x, velocity_y))
			nextState = nextState.astype(int)
			intersection = self.projectedIntersection(nextState)
			if (intersection == 2):
				nextState = np.append(copy.copy(
								self.start[np.random.randint(self.start.shape[0])]), [0,0])
				reward = -1
			elif (intersection == 1):
				# episode = np.vstack((episode, np.append(nextState,np.array((0, action[0], action[1])))))
				self.finishState[0,2] = velocity_x
				self.finishState[0,3] = velocity_y
				break
			else:
				reward = -1
			# print(np.append(previousState,np.array((reward, actionIndex))))

			if(episode.shape[0] == 1):
				episode = np.append(previousState,np.array((reward, actionIndex)))
			else:
				episode = np.vstack((episode, 
							np.append(previousState,np.array((reward, actionIndex)))))
			previousState = copy.copy(nextState)
		
		episode = np.vstack((episode, 
							np.append(previousState,np.array((reward, actionIndex)))))
		# print(np.append(previousState,np.array((reward, actionIndex))))
		episode = np.vstack((episode, 
							np.append(self.finishState,np.array((0, actionIndex)))))
		# print(np.append(self.finishState,np.array((reward, actionIndex))))
		return episode

	def projectedIntersection(self, nextState):
		projectedState = copy.copy(nextState[0:2])
		for i in range(0, max(nextState[2], nextState[3])+1):
			if(np.amin(np.sum(np.absolute(self.finish - np.array(projectedState[0:2])),axis=1)) ==0):
				self.finishState[0,0] = copy.copy(projectedState[0])
				self.finishState[0,1] = copy.copy(projectedState[1])
				return 1
			if(projectedState[0] < 0 or projectedState[0] >= self.map.shape[0]):
				return 2
			elif(projectedState[1] < 0 or projectedState[1] >= self.map.shape[1]):
				return 2
			elif(self.map[projectedState[0],projectedState[1]] == 1):
				return 2
			projectedState[0] = max(0,projectedState[0]-1)
			projectedState[1] = max(0,projectedState[1]-1)
		return 3
	def stateIndex(self, currentState):
		return int(36*(currentState[1]*self.map.shape[0] + currentState[0]))

if __name__ == '__main__':
	mc = incrementalOffPolicy(filename = 'track1.txt')
	gamma = 0.5
	count = 0
	otherCount = 0
	while count < 10000:
		print("count: ", count, "  otherCount: ", otherCount)
		episode = mc.generateEpisode()
		# print(mc.finish)
		# print(episode)
		# state : {x, y, vx, vy}
		# action-value function q(every state): vector of 36*9 elements for every grid
		G = 0
		W = 1
		otherCount = 0
		for i in range(episode.shape[0]-2, -1, -1):
			G = gamma*G + episode[i,4]
			ind1 = mc.stateIndex(episode[i])
			ind2 =  int(copy.copy(episode[i,5]))
			# print(ind1, ind2)
			mc.C[ind1,ind2] = mc.C[ind1,ind2] + W
			mc.Q[ind1,ind2] = mc.Q[ind1,ind2] + W/mc.C[ind1,ind2]*(G - mc.Q[ind1,ind2])
			targetAction = mc.targetPolicy(episode[i])
			print(targetAction-mc.PossibleActions[ind2])
			if np.sum(np.absolute(targetAction-mc.PossibleActions[ind2])) == 0:
				targetPI = 1
			else:
				targetPI = 0
			W = W*targetPI/(1/9)
			if W == 0:
				break
			otherCount += 1
		count += 1
	np.savetxt('q.txt', mc.Q, fmt='%.18e', delimiter=' ', newline='\n')