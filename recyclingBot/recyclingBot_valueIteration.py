import copy

import numpy as np
import matplotlib.pyplot as plt

alpha = 0.7
beta = 0.5
rSearch = 2
rWait = 1
discount = np.linspace(0.1, 1, 20, endpoint=False)

# Initialization
initialValue = [0, 0]

iteration = np.empty(shape=(1,))
value = np.empty(shape=(2,))
# Value Iteration
for i in range(0,discount.size):
	itr = 0
	newValue = copy.copy(initialValue)
	termCond = 10
	dis = discount[i]
	while (termCond > 10**(-4)):
		
		oldValue = copy.copy(newValue)
		newValue[1] = max((rSearch + dis*(alpha*oldValue[1] + (1 - alpha)*oldValue[0])),
						(rWait + dis*oldValue[1]))
		newValue[0] = max((beta*rSearch - 3*(1-beta) + dis*((1-beta)*oldValue[1] + beta*oldValue[0])),
						(rWait + dis*oldValue[0]), (dis*oldValue[1]))
		termCond = max((newValue[1] - oldValue[1]), (newValue[0] - oldValue[0]))
		itr += 1
		print(itr, termCond, oldValue, newValue)
	iteration = np.hstack((iteration, np.array(itr)))
	# for j in range(1,len(newValue)):
	value = np.vstack((value, np.array(newValue)))

final_index = discount.size + 1
xAxis = np.linspace(0.1, 1, 20, endpoint=False)
plt.figure(1)
plt.plot(xAxis, iteration[1:final_index], linewidth=3,label='iteration',color='b')
plt.title('Iterations wrt Discount Factor')
plt.legend()
plt.ylabel('Iterations')
plt.xlabel('Discount Factor')
plt.grid()

plt.figure(2)
plt.plot(xAxis, np.transpose(value)[0][1:final_index], linewidth=3,label='state_high',color='r')
plt.plot(xAxis, np.transpose(value)[1][1:final_index], linewidth=3,label='state_low',color='g')
plt.title('Value wrt Discount Factor')
plt.legend()
plt.ylabel('Value')
plt.xlabel('Discount Factor')
plt.grid()
plt.show()