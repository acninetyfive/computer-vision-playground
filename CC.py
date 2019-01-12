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
					equivalences[x] = component
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


		'''for x in np.argwhere(labels):
			if equivalences[labels[x[0]][x[1]]]:
				labels[x[0]][x[1]] = min(equivalences[labels[x[0]][x[1]]])
		'''
		for i in range(len(img)):
			for j in range(len(img[0])):
				if (labels[i][j] and equivalences[labels[i][j]]):
					labels[i][j] = min(equivalences[labels[i][j]])

		if(self.show_timer): print("second pass:", '%.6f'%(time.perf_counter() - t))

		if(self.show_timer): print("two_pass_cc:", '%.6f'%(time.perf_counter() - s))
		return labels

def change_component_with_key(key, fig, components, colored_img, original):
	
	global img_counter

	if (key == "left" and img_counter > 0):
		img_counter -= 1
		if (img_counter == len(components) - 1):

			fig.canvas.set_window_title("Component # " + str(img_counter))

			plt.subplot(1,3,1)
			plt.imshow(original)
			plt.axis('off')

			plt.subplot(1, 3, 2)
			plt.imshow(colored_img)
			plt.axis('off')

			plt.subplot(1, 3, 3)
			plt.imshow(components[img_counter])
			plt.axis('off')

			plt.draw()

			return

		
	elif (key == "right" and img_counter < len(components)):
		img_counter += 1

	if(img_counter < len(components)):
		
		fig.canvas.set_window_title("Component # " + str(img_counter))

		plt.imshow(components[img_counter])
		plt.draw()

		#img_counter += 1

	else:

		cc_count = len(components)

		fig.clear()
		fig.canvas.set_window_title("All " + str(cc_count) + " Components")

		rows = math.floor(math.sqrt(cc_count + 2))
		cols = math.ceil((cc_count + 2)/rows)


		for i in range(cc_count):
			plt.subplot(rows, cols, i + 1)
			plt.imshow(components[i])
			plt.axis('off')


		plt.subplot(rows, cols, cc_count + 1)
		plt.imshow(colored_img)
		plt.axis('off')

		plt.subplot(rows, cols, cc_count + 2)
		plt.imshow(original)
		plt.axis('off')

		plt.draw()

	

def rgb_to_ccs_driver(file, verbose = False, min_size = -1):
	s = time.perf_counter()

	cc_im = CC_Image(file, verbose)

	if (min_size == -1):
		min_size = .0001 * cc_im.original.size / 3
		print("No minimum component size given, defaulting to .01%: " + '%.6f'%(min_size) + " pixels")

	gray_img = cc_im.rgb_to_gray(deepcopy(cc_im.original))
	binary_img = cc_im.gray_to_binary(gray_img)
	labels = cc_im.remove_small_ccs(cc_im.two_pass_cc(binary_img), min_size)
	colored_img, color_dict = cc_im.color_all_ccs(labels)


	plt.subplot(2,2,1)
	plt.imshow(cc_im.original)
	plt.axis('off')

	plt.subplot(2,2,2)
	plt.imshow(gray_img, cmap = "gray")
	plt.axis('off')

	plt.subplot(2,2,3)
	plt.imshow(binary_img, cmap = "gray")
	plt.axis('off')

	plt.subplot(2,2,4)
	plt.imshow(colored_img)
	plt.axis('off')

	print("Full runtime:", '%.6f'%(time.perf_counter() - s))
	print("Found " + str(cc_im.number_of_ccs(labels)) + " connected components")
	plt.show()
	return

def highlight_each_cc_driver(file, verbose = False, min_size = -1):
	s = time.perf_counter()

	cc_im = CC_Image(file, verbose)

	if (min_size == -1):
		min_size = .0001 * cc_im.original.size / 3
		print("No minimum component size given, defaulting to .01%: " + str(min_size) + " pixels")

	gray_img = cc_im.rgb_to_gray(deepcopy(cc_im.original))
	binary_img = cc_im.gray_to_binary(gray_img)
	labels = cc_im.remove_small_ccs(cc_im.two_pass_cc(binary_img), min_size)
	colored_img, color_dict = cc_im.color_all_ccs(labels)

	cc_count = cc_im.number_of_ccs(labels)
	

	unique_labels = np.unique(labels)

	ul_len = len(unique_labels)

	t = time.perf_counter()

	components = []

	for i, x in enumerate(np.delete(unique_labels, np.argwhere(unique_labels == 0))):

		img = cc_im.highlight_one_cc(labels, colored_img, x)
		components.append(img)

	global img_counter

	fig = plt.figure()
	fig.canvas.set_window_title("Component # " + str(img_counter))
	fig.canvas.mpl_connect('key_press_event', lambda event: change_component_with_key(event.key, fig, components, colored_img, cc_im.original))

	print("Highlight all CC's:", '%.6f'%(time.perf_counter() - t))

	plt.subplot(1,3,1)
	plt.imshow(cc_im.original)
	plt.axis('off')

	plt.subplot(1, 3, 2)
	plt.imshow(colored_img)
	plt.axis('off')

	plt.subplot(1, 3, 3)
	plt.imshow(components[img_counter])
	plt.axis('off')

	#img_counter += 1

	print("Full runtime:", '%.6f'%(time.perf_counter() - s))
	print("Found " + str(cc_count) + " connected components")

	plt.show()
	return

def print_help_message():
	print("Usage Format: py CC.py file.jpg [flags]")
	print()
	print()
	print("Valid flags:")
	print()
	print("-draw_each_cc:")
	print("\tReturns pyplot with each component drawn separately as well as together. Use arrow keys to navigate.")
	print("\tWhen absent, pyplot will show original, grayscale, binary, and component colored images in one view.")
	print()
	print("-min_size [number]")
	print("\tSet the minimum size for a component to be counted.")
	print("\tDefault is set to 0.01% " + "of image size. To include all, set to 0.")
	print()
	print("-verbose")
	print("\tPrints run times for each function.")
	print()

def parse_runtime_args():
	valid_flags = ["-verbose", "-min_size", "-draw_each_cc", "-help", "-h"]

	print("For help and information, use '-h' or '-help' flag")
	print()

	if (len(sys.argv) == 0):
		print("No image selected, exiting")
		return -1

	if (sys.argv[1] == "-h" or sys.argv[1] == "-help"):
		print_help_message()
		return 1

	if (not sys.argv[1].endswith(".jpg") and not sys.argv[1].endswith(".bmp")):
		print("Must enter .jpg file as first argument")
		return -1



	runtime_arg_dict = {"file":sys.argv[1], "-verbose": False, "-draw_each_cc": False, "-min_size": -1}

	for i in range(2, len(sys.argv)):
		if (sys.argv[i] in valid_flags):
			if (sys.argv[i] == "-h" or sys.argv[i] == "-help"):
				print_help_message()

			if (sys.argv[i] == "-min_size"):
				try:
					runtime_arg_dict["-min_size"] = int(sys.argv[i+1])
				except ValueError:
					print("error, min_size entered is not a number")
					return -1
				except IndexError:
					print("error, no min_size entered")
					return -1
			else:
				runtime_arg_dict[sys.argv[i]] = True

	if (runtime_arg_dict["-draw_each_cc"]):
		highlight_each_cc_driver(runtime_arg_dict["file"], runtime_arg_dict["-verbose"], int(runtime_arg_dict["-min_size"]))
	else:
		rgb_to_ccs_driver(runtime_arg_dict["file"], runtime_arg_dict["-verbose"], int(runtime_arg_dict["-min_size"]))	


	return 1


img_counter = 0

runtime_arg_dict = parse_runtime_args()
'''
cc_im = CC_Image(sys.argv[1])

img = cc_im.gray_to_binary(cc_im.rgb_to_gray(cc_im.original))

label_counter = 1

labels = np.zeros(img.shape)
equivalences = {}

for i in range(len(img)):
	for j in range(len(img[0])):
		if (img[i][j]):
			neighbors = cc_im.get_neighbors(labels, i, j, 4)
			if (neighbors):
				labels[i][j] = min(neighbors)
				for n in neighbors:
					equivalences[n].update(set(neighbors))
			else:
				labels[i][j] = label_counter
				equivalences[label_counter] = set()
				label_counter += 1

print(equivalences)
equivalences = cc_im.merge_equivalences(equivalences)

print(equivalences)
'''