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

	for t in range (time-1):
		u = np.expand_dims(input[t,:], axis=1)
		x = (1-a)*x + a*np.tanh(np.dot(W, x) + np.dot(W_in, u) + v + xi)
		x = f(x)
		if t > washout:
			M[t - washout, :] = np.squeeze(x)
			T[t - washout, :] = teacher[t+1, :]

	if mp == True:
		# Moore-Penrose Pseudoinverse
		W_out = np.dot(np.linalg.pinv(M), T).T
	else:
		# Ridge Regression
		W_out = np.dot(np.dot(T.T, M), np.linalg.inv(np.dot(M.T,M) + beta*np.eye(N)))

	return W_out, M

def f (x):
	'''
				r_i = r_i; if i is odd
		f(r) = 
				r_i = r_i^2; if i is even

	'''
	x[1::2] *= x[1::2]
	return x

def run(W, W_in, W_out, N, K, L, input, a, xi, M, time, c):
	'''
		W: 		reservoir matrix
		W_in: 	input matrix
		W_out: 	trained output matrix
		N: 		number of neurons in the reservoir
		K:      input dimension
		L: 		output dimension
		input: 	input data
		a:      leaking rate
		xi:     addition constant ~ 0
		M:      state collection matrix
		time:   length of input data
		c:      coupling parameter

	'''
	if input is None:
		input = np.zeros((time, K))

	x = np.expand_dims(M[-1, :], axis=1)
	y = np.zeros((L, 1))
	states = np.zeros((time, N))
	outputs = np.zeros((time, L))
	u = np.expand_dims(input[1,:], axis=1)

	for t in range(1, time):
		if ((t < 20)):
			u = np.expand_dims(input[t,:], axis=1)

		# Coupling - input module
		u_prime = np.expand_dims(input[t,:], axis=1)
		u = u + c * (u_prime - u)

		# forward pass
		x = (1-a)*x + a*np.tanh(np.dot(W, x) + np.dot(W_in, u) + xi)
		u = np.dot(W_out, f(x))
		states[t, :] = np.squeeze(x)
		outputs[t,:] = np.squeeze(u)

	return outputs

K = 3
N = 600
L = 3

washout = 5
trainLen = 300
testLen = 2500
a = 1
v = 0
xi = 0

sigma = 0.1
d_prime = 0.3
rho = 1.2
coupling = 0.99

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

test_inputs = np.asarray([x_t[trainLen+1:trainLen + testLen + 1], y_t[trainLen+1:trainLen + testLen + 1], z_t[trainLen+1:trainLen + testLen + 1]])
test_inputs = test_inputs.T

results = run(W, W_in, W_out, N, K, L, test_inputs, a, v, M, testLen-1, coupling)

plt.figure()
plt.plot(results[:,0])
plt.plot(x_t[trainLen+1:trainLen+testLen+1])


plt.figure()
plt.plot(results[:,1])
plt.plot(y_t[trainLen+1:trainLen+testLen+1])

plt.figure()
plt.plot(results[:,2])
plt.plot(z_t[trainLen+1:trainLen+testLen+1])

plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(results[:,0], results[:,1], results[:,2])
plt.show()
