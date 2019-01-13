import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
import numpy as np
import math
import random
import time
import sys
from copy import deepcopy
from skimage import filters


class CC_Image:
	def __init__(self, img, st = False):
		s = time.perf_counter()
		self.show_timer = int(st)
		self.original = plt.imread(img)
		self.original.setflags(write=1)

		if(self.show_timer):	print("init:", '%.6f'%(time.perf_counter() - s))


	def color_all_ccs(self, labels):
		s = time.perf_counter()

		#labels = self.two_pass_cc(input_image)

		colored_img = deepcopy(self.original)

		color_dict = {}

		ccs = np.unique(labels)

		n = len(ccs)
		colors = self.get_rand_colors(len(np.unique(labels)))
		for i in range(n):
			color_dict[ccs[i]] = colors[i]

		for i in range(len(colored_img)):
			for j in range(len(colored_img[0])):
				if (labels[i][j]):
					colored_img[i][j] = color_dict[labels[i][j]]
				else:
					colored_img[i][j] = [0,0,0]


		if(self.show_timer): print("color_all_ccs:", '%.6f'%(time.perf_counter() - s))
		return colored_img, color_dict


	def draw_rgb_to_binary(self):
		s = time.perf_counter()

		plt.subplot(1,3,1)
		plt.imshow(self.original)

		plt.subplot(1,3,2)
		gray_img = self.rgb_to_gray(deepcopy(self.original))
		plt.imshow(gray_img, cmap = "gray")

		plt.subplot(1,3,3)
		plt.imshow(self.gray_to_binary(gray_img), cmap = "gray")

		if(self.show_timer): print("draw_rgb_to_binary:", '%.6f'%(time.perf_counter() - s))
		return


	def draw_rgb_to_ccs(self):
		s = time.perf_counter()

		plt.subplot(2,2,1)
		plt.imshow(self.original)

		plt.subplot(2,2,2)
		gray_img = self.rgb_to_gray(deepcopy(self.original))
		plt.imshow(gray_img, cmap = "gray")

		plt.subplot(2,2,3)
		binary_img = self.gray_to_binary(gray_img)
		plt.imshow(binary_img, cmap = "gray")

		plt.subplot(2,2,4)
		colored_img = self.color_all_ccs(binary_img)
		plt.imshow(colored_img)

		if(self.show_timer): print("draw_rgb_to_ccs:", '%.6f'%(time.perf_counter() - s))
		return colored_img


	def get_neighbors(self, labels, i, j, n):
		s = time.perf_counter()

		neighbors = []
		if (n != 4 and n != 8):
			n = 8

		#always add 4-connectivity points
		if (i > 0 and labels[i-1][j]):
			neighbors.append(labels[i-1, j])
		if (j > 0 and labels[i][j-1]):
			neighbors.append(labels[i, j-1])
		if (i < len(labels) - 1 and labels[i+1][j]):
			neighbors.append(labels[i+1, j])
		if (j < len(labels[0]) - 1 and labels[i][j+1]):
			neighbors.append(labels[i, j+1])

		#then add 8-connectivity points if desired
		if n == 8:
			if (i > 0 and j > 0 and labels[i-1][j-1]):
				neighbors.append(labels[i-1, j-1])
			if (i > 0 and j < len(labels[0]) - 1 and labels[i-1][j+1]):
				neighbors.append(labels[i-1, j+1])
			if (i < len(labels) - 1 and j > 0 and labels[i+1][j-1]):
				neighbors.append(labels[i+1, j-1])
			if (i < len(labels) - 1 and j < len(labels[0]) - 1 and labels[i+1][j+1]):
				neighbors.append(labels[i+1, j+1])

		#if(self.show_timer): print("get_neighbors:", '%.6f'%(time.perf_counter() - s))
		return neighbors


	def get_rand_colors(self, n):
		s = time.perf_counter()

		r = lambda: random.randint(0,255)
		
		colors = [[r(),r(),r()] for i in range(n)]

		color_dict = {}


		#if(self.show_timer): print("get_rand_colors:", '%.6f'%(time.perf_counter() - s))
		return colors


	def gray_to_binary(self, img):
		return img > filters.threshold_yen(img)


	def highlight_one_cc(self, labels, img, cc):
		s = time.perf_counter()

		hl_img = np.full(img.shape, [0,0,0])

		for x in np.argwhere(labels != 0):
			hl_img[x[0]][x[1]] = [200,200,200]

		for x in np.argwhere(labels == cc):
			hl_img[x[0]][x[1]] = img[x[0]][x[1]]

		


		#if(self.show_timer): print("highlight_one_cc:", '%.6f'%(time.perf_counter() - s))
		return hl_img


	def merge_equivalences(self, equivalences):
		s = time.perf_counter()

		merged = set()

		component = set()
		visited = set()

		for e in equivalences.keys():
			if (e not in visited):
				visited.add(e)
				component.add(e)

				q = list(equivalences[e])

				while len(q) > 0:
					n = q.pop(0)
					visited.add(n)
					component.add(n)
					for x in equivalences[n]:
						if x not in visited:
							q.append(x)
				for x in component:
					equivalences[x] = min(component)
				component = set()

		if(self.show_timer): print("merge_equivalences:", '%.6f'%(time.perf_counter() - s))
		return equivalences




	def number_of_ccs(self, labels):
		s = time.perf_counter()

		count = len(np.unique(labels))

		#if(self.show_timer): print("number_of_ccs:", '%.6f'%(time.perf_counter() - s))
		return count - 1 #remove the count for the "background" component


	def remove_small_ccs(self, labels, n):
		s = time.perf_counter()

		vals, counts = np.unique(labels, return_counts = True)

		count_dict = dict(zip(vals, counts))

		masked = set([i for i in count_dict if (count_dict[i] < n)])

		for i in range(len(labels)):
			for j in range(len(labels[0])):
				if labels[i][j] in masked:
					labels[i][j] = 0

		if(self.show_timer): print("remove_small_ccs:", '%.6f'%(time.perf_counter() - s))
		return labels


	def rgb_to_gray(self, img):
		s = time.perf_counter()
		gs = [0.2126, 0.7152, 0.0722]
		img = np.dot(img, gs)

		if(self.show_timer): print("rgb_to_gray:", '%.6f'%(time.perf_counter() - s))
		return img
	

	def two_pass_cc(self, img):
		s = time.perf_counter()

		label_counter = 1

		labels = np.zeros(img.shape)
		equivalences = {}

		t = time.perf_counter()

		for i in range(len(img)):
			for j in range(len(img[0])):
				if (img[i][j]):
					neighbors = self.get_neighbors(labels, i, j, 4)
					if (neighbors):
						labels[i][j] = min(neighbors)
						for n in neighbors:
							equivalences[n].update(set(neighbors))
					else:
						labels[i][j] = label_counter
						equivalences[label_counter] = set()
						label_counter += 1

		if(self.show_timer): print("first pass:", '%.6f'%(time.perf_counter() - t))

		equivalences = self.merge_equivalences(equivalences)

		t = time.perf_counter()


		for i in range(len(img)):
			for j in range(len(img[0])):
				if (labels[i][j]):
					labels[i][j] = equivalences[labels[i][j]]

		if(self.show_timer): print("second pass:", '%.6f'%(time.perf_counter() - t))

		if(self.show_timer): print("two_pass_cc:", '%.6f'%(time.perf_counter() - s))
		return labels
