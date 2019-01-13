import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import math
import random
import time
import scipy.spatial.distance as sd
import sys
from copy import deepcopy



class K_Means:
	def __init__(self, img, means_num):
		s = time.perf_counter()
		self.initialize_means(means_num)
		self.original = plt.imread(img)
		self.original.setflags(write=1)
		self.points = plt.imread(img) #points is an array p[x][y][r, b, g]
		self.points.setflags(write=1)
		self.point_mean_dict = {} #key is (x,y), value is [closest mean]
		
		for x in range(len(self.points)):
			for y in range(len(self.points[0])):
				self.point_mean_dict[(x,y)] = [None]
		print("init:", time.perf_counter() - s)
		
	def distance(self, p1, p2): #p1 and p2 are arrays (vectors)
		if len(p1) == 0 or len(p1) != len(p2):
			print("ERR", p1, p2, self.means)
			return "Invalid points"
		d_squared = 0
		for i in range(len(p1)):
			d_squared += (p1[i] - p2[i]) ** 2
		return math.sqrt(d_squared)
		
	def initialize_means(self, k):
		self.means = None
		
		r = lambda: random.randint(0,255)
		
		self.means = [[r(),r(),r()] for x in range(k)]
	
	def set_means(self, m_list):
		self.means = m_list
	
	def separate_points(self):
		s = time.perf_counter()
		
		for x in range(len(self.points)):
			dist_matrix = sd.cdist(self.points[x], self.means)
			for y in range(len(dist_matrix)):
				self.point_mean_dict[(x,y)] = dist_matrix[y].argmin()
		
		#print("separate:", time.perf_counter() - s)
		return self.point_mean_dict
		
	def recenter(self, c): 
		s = time.perf_counter()
		new_center = self.get_center([p for p in self.get_points_by_center(c)])
		if new_center == self.means[c]:
			#print("recenter", time.perf_counter() - s)
			return False #no change
		elif len(new_center) != 0:
			self.means[c] = new_center
			#print("recenter", time.perf_counter() - s)
			return True #change made
		
	def get_center(self, pts_lst):
		s = time.perf_counter()
		l = [sum(x)/len(x) for x in zip(*pts_lst)]
		#print("get_center", time.perf_counter() - s)
		return l
		
	def get_points_by_center(self, c): #c is the index of the desired center in self.means
		s = time.perf_counter()
		l = [self.points[p[0]][p[1]] for p in self.point_mean_dict if self.point_mean_dict[p] == c]
		#print("get_points_by_center", time.perf_counter() - s)
		return l 
		
	def recolor(self):
		for x in range(len(km.points)):
			for y in range(len(km.points[0])):
				km.points[x][y] = km.means[km.point_mean_dict[(x,y)]]
		return
		
	def step(self):
		s= time.perf_counter()
		self.separate_points()
		change = False
		for i in range(len(self.means)):
			if self.recenter(i) == True:
				change = True
		print("step", time.perf_counter() - s)
		return change
		
	def full_run(self):
		t = time.perf_counter()
		s = 0
		while self.step():
			s += 1
		print("full_run", time.perf_counter() - t)
		return s

	def ss_err(self):
		sse = 0
		for i in range(len(self.means)):
			p = self.get_points_by_center(i)
			for x in p:
				sse += self.distance(x,self.means[i])
		return sse
		
	def test_k_range(self, start, stop):
		err_lst = None
		if stop > len(self.points) + 1:
			stop = len(self.points) + 1
		for i in range(start, stop):
			self.initialize_means(i)
			self.full_run()
			if err_lst == None:
				err_lst = [self.ss_err()]
			else:
				err_lst.append(self.ss_err())
		return np.array(err_lst)
		
	def draw_elbow_graph(self, start, stop):
		x = np.array([x for x in range(start,stop)])
		y = self.test_k_range(start,stop)
		plt.scatter(x,y)
		plt.show()
		return	

	def show_n_runs(self, n):
		n += 1
		rows = math.floor(math.sqrt(n))
		cols = math.ceil(n/rows)
		print(rows, cols)
		
		for i in range(1, n):
			self.initialize_means(i)
			print("K =", i)
			self.points = deepcopy(self.original)
			self.full_run()
			self.recolor()
			plt.subplot(rows, cols, i)
			plt.imshow(self.points)		
		plt.subplot(rows, cols, n)
		plt.imshow(self.original)
		
		plt.show()
		return
	



km = K_Means(sys.argv[1], int(sys.argv[2]))

obama = [[215, 24, 32],[0, 50, 77],[252, 228, 168 ],[113, 150, 159]] 
gbw = [[161, 208, 128], [73, 72, 88], [232, 231, 249]]

#km.set_means(obama)

#km.separate_points()

#km.draw_elbow_graph(2,12)

#km.show_n_runs(int(sys.argv[2]))

i = int(sys.argv[2])


km.initialize_means(i)
print("K =", i)
km.points = deepcopy(km.original)
km.full_run()
km.recolor()
plt.subplot(1, 2, 1)
plt.imshow(km.points)		
plt.subplot(1,2,2)
plt.imshow(km.original)

plt.show()






