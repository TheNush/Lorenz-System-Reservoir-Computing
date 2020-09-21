import numpy as np
import networkx as nx
import random
from scipy import linalg
import pandas as pd
import matplotlib.pyplot as plt

def reservoir_generator(N, rho, d_w, sigma):
	'''
		N: number of neurons in the reservoir
		rho: desired spectral radius of the reservoir
		d_w: density of the reservoir matrix

	'''
	done = False
	while (done == False):
		# generate N x N graph, with d_w proportion of entries
		G = nx.gnp_random_graph(N, d_w, directed=True)
		W = np.asarray(nx.to_numpy_matrix(G))

		for i in range(N):
			for j in range(N):
				if W[i][j] == 1.0:
					#sample a random number from (-sigma, sigma) and place it in W(i,j)
					W[i][j] = np.random.uniform(low=-sigma, high=sigma)

		# set spectral radius
		lambdaa = max(abs(linalg.eigvals(W)))
		if (lambdaa != 0.0):
			done = True
		else:
			continue
		print("Current spectral radius: ", lambdaa)
		print("Changing spectral radius to ", rho)
		W *= rho / lambdaa
		print("Done.")

	return W

def input_matrix_generator(N, K, d_in, sigma_in):
	'''
		N: number of neurons in the reservoir
		K: number of inputs
		d_in: input matrix density
		sigma_in: desied input radius

	'''
	W_in = np.zeros((N,K))
	for i in range(N):
		for j in range(K):
			# sample a random number from [0,1)
			p = random.random()
			if p < d_in:
				W_in[i][j] = np.random.uniform(-sigma_in, sigma_in)

	return W_in

def train(W, W_in, N, K, L, input, teacher, washout, a, v, xi, mp=False, beta=1e-8):
	time = len(teacher)
	if input is None:
		input = np.zeros((time, K))

	M = np.zeros((time - washout, N))
	T = np.zeros((time - washout, L))
	x = np.zeros((N,1))
	y = np.zeros((L,1))

	for t in range (time):
		u = np.expand_dims(input[t,:], axis=1)
		x = (1-a)*x + a*np.tanh(np.dot(W, x) + np.dot(W_in, u) + v + xi)
		if t > washout:
			M[t - washout, :] = np.squeeze(x)
			T[t - washout, :] = teacher[t, :]

		# y = teacher[t, :]

	if mp == True:
		# Moore-Penrose Pseudoinverse
		W_out = np.dot(np.linalg.pinv(M), T).T
	else:
		# Ridge Regression
		W_out = np.dot(np.dot(T.T, M), np.linalg.inv(np.dot(M.T,M) + beta*np.eye(N)))

	return W_out, M

def run(W, W_in, W_out, N, K, L, input, a, xi, M, time):
	if input is None:
		input = np.zeros((time, K))

	x = np.expand_dims(M[-1, :], axis=1)
	y = np.zeros((L, 1))
	states = np.zeros((time, N))
	outputs = np.zeros((time, L))

	for t in range(1, time):
		u = np.expand_dims(input[t,:], axis=1)
		x = (1-a)*x + a*np.tanh(np.dot(W, x) + np.dot(W_in, u) + xi)
		y = np.dot(W_out, x)
		states[t, :] = np.squeeze(x)
		outputs[t,:] = np.squeeze(y)

	return outputs

# W = reservoir_generator(10, 1.2, 0.3, 0.1)
# W_in = input_matrix_generator(10, 3, 0.3, 1)

# W_out, M = train(W, W_in, N=10, K=3, L=3, input=None, teacher=np.zeros((20,3)), washout=1, a=1, v=0, xi=0)

# outputs = run (W, W_in, W_out, 10, 3, 3, None, 1, 0, M, 20)

# K = 1
# N = 3
# L = 1

# washout = 50
# a = 1
# v = 0
# xi = 0

# W = reservoir_generator(N, 0.9, 0.5, 1)
# W_in = input_matrix_generator(N, K, 1, 1)

# # W = np.asarray([[0, -0.86, 0],[-0.72, 0, -0.40], [1.04, 0, 0]])

# # W_in = np.asarray([[-0.49],[0.19], [0.57]])

# input_path = "C:/Users/dhanu/Desktop/Work/Research/Simulate Chaotic Systems/Sine/input.xlsx"
# output_path = "C:/Users/dhanu/Desktop/Work/Research/Simulate Chaotic Systems/Sine/output.xlsx"
# inputs = pd.read_excel(input_path, header=None)
# outputs = pd.read_excel(output_path, header=None)
# inputs = np.expand_dims(inputs.loc[:,0].to_numpy(), axis=1)
# outputs = np.expand_dims(outputs.loc[:,0].to_numpy(), axis=1)

# W_out, M, T = train(W, W_in, N, K, L, inputs, outputs, washout, a, v, xi, mp=True)

# results = run(W, W_in, W_out, N, K, L, inputs[:100,:], a, v, M, 100)

# plt.figure()
# plt.plot(results)
# plt.plot(outputs[:100,:])
# plt.show()

K = 3
N = 600
L = 3

washout = 100
trainLen = 2500
testLen = 2500
a = 1
v = 0
xi = 0

sigma = 0.1
d_prime = 0.3
rho = 1.2

data_path = "C:/Users/dhanu/Desktop/Work/Research/Simulate Chaotic Systems/Lorenz System/Lorenz-deltaT=0.01.xlsx"
data = pd.read_excel(data_path, header=None)

x_t = data.loc[:, 0].to_numpy()
y_t = data.loc[:, 1].to_numpy()
z_t = data.loc[:, 2].to_numpy()

inputs = np.asarray([x_t[:trainLen], y_t[:trainLen], z_t[:trainLen]])
inputs = inputs.T
outputs = inputs

W = reservoir_generator(N, rho, d_prime, sigma)
W_in = input_matrix_generator(N, K, 1, sigma)

W_out, M = train(W, W_in, N, K, L, inputs, outputs, washout, a, v, xi)

test_inputs = np.asarray([x_t[trainLen+1:], y_t[trainLen+1:], z_t[trainLen+1:]])
test_inputs = test_inputs.T

results = run(W, W_in, W_out, N, K, L, test_inputs, a, v, M, 2499)

plt.figure()
plt.plot(results[:,0])
plt.plot(x_t[trainLen+1:])


plt.figure()
plt.plot(results[:,1])
plt.plot(y_t[trainLen+1:])

plt.figure()
plt.plot(results[:,2])
plt.plot(z_t[trainLen+1:])

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(results[:,0], results[:,1], results[:,2])
plt.show()
