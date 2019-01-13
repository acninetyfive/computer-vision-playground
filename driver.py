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

import CC 
#import k_means_image #not utilized yet








	#
	#Utility 
	#Functions
	#


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


	#
	#Driver
	#Functions
	#

	
def highlight_each_cc_driver(file, verbose = False, min_size = -1):
	s = time.perf_counter()

	cc_im = CC.CC_Image(file, verbose)

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

def rgb_to_ccs_driver(file, verbose = False, min_size = -1):
	s = time.perf_counter()

	cc_im = CC.CC_Image(file, verbose)

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


img_counter = 0

runtime_arg_dict = parse_runtime_args()
