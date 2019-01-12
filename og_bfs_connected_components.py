import numpy as np

test_matrix = [[0,0,0,0],
				[1,1,0,0],
				[1,0,2,2],
				[0,0,0,0]]
small = [[0,0],[1,1]]


def bfs(node, visited, matrix):
	visited.add(node)

	q = get_adjacent(node, matrix)

	counter = 0
	while len(q) > 0:
		#print("bfs loop #: " + str(counter))
		counter += 1
		n = q.pop(0)
		#print(n)
		visited.add(n)
		for x in get_adjacent(n, matrix):
			if x not in visited:
				q.append(x)
		#print("queue: " + str(q))
		#print("visited: " + str(visited))
		#print()
	return visited


def get_adjacent(node, matrix):
	#print(matrix[node[0]][node[1]])
	adj = []
	val = matrix[node[0]][node[1]]
	if (node[0] > 0 and np.all(matrix[node[0]-1][node[1]] == val)):
		adj.append((node[0]-1, node[1]))
	if (node[0] < len(matrix) - 1 and np.all(matrix[node[0]+1][node[1]] == val)):
		adj.append((node[0]+1, node[1]))
	if (node[1] > 0 and np.all(matrix[node[0]][node[1]-1] == val)):
		adj.append((node[0], node[1]-1))
	if (node[1] < len(matrix[0]) - 1 and np.all(matrix[node[0]][node[1] + 1] == val)):
		adj.append((node[0], node[1]+1))
	#print("node: " + str(node))
	#print("adj: " + str(adj))

	return adj


def connected_components(matrix):
	cc = []
	nodes = set(np.ndindex((len(matrix),len(matrix[0]))))
	c = 0
	while nodes:
		c += 1
		#print ("cc counter " + str(c)) 
		visited = set()
		component = bfs(nodes.pop(), set(), matrix)
		cc.append(component)
		#print("component: " + str(component))
		#print("length: " + str(len(component)))
		nodes = nodes - component
		#print("# of nodes left: " + str(len(nodes)))
		#input()

	return cc

def highlight_cc(cc, matrix, low, high):
	hlm = [[low for j in range(len(matrix[0]))] for i in range(len(matrix))]
	for x in cc:
		hlm[x[0]][x[1]] = high
	return hlm

#print(bfs((3,0), visited, set(), test_matrix))

for x in test_matrix:
	print(x)

print()

for c in connected_components(test_matrix):
	for l in highlight_cc(c, test_matrix, "-", "O"):
		print(l)
	print()


