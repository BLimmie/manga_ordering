from collections import defaultdict

from .metrics import calculate_metrics_list


class Graph:
    """
    The code for this class is based on geeksforgeeks.com
    """

    def __init__(self, vertices):
        self.graph = defaultdict(list)
        self.V = vertices

    def addEdge(self, u, v):
        self.graph[u].append([v])

    def topologicalSortUtil(self, v, visited, stack):

        visited[v] = True

        for i in self.graph[v]:
            if not visited[i[0]]:
                self.topologicalSortUtil(i[0], visited, stack)

        stack.insert(0, v)

    def topologicalSort(self):
        visited = [False] * self.V
        stack = []

        for i in range(self.V):
            if not visited[i]:
                self.topologicalSortUtil(i, visited, stack)

        return stack

    def isCyclicUtil(self, v, visited, recStack):

        visited[v] = True
        recStack[v] = True

        for neighbour in self.graph[v]:
            if not visited[neighbour[0]]:
                if self.isCyclicUtil(
                        neighbour[0], visited, recStack):
                    return True
            elif recStack[neighbour[0]]:
                self.graph[v].remove(neighbour)
                return True

        recStack[v] = False
        return False

    def isCyclic(self):
        visited = [False] * self.V
        recStack = [False] * self.V
        for node in range(self.V):
            if not visited[node]:
                if self.isCyclicUtil(node, visited, recStack):
                    return True
        return False


def convert_to_graph(logits, positions):
    # get no vertices (len logits = n(n-1)/2
    nvert = int((2 * len(logits)) ** 0.5)+1
    # create graph obj
    g = Graph(nvert)

    # read pred label
    for logit, pos in zip(logits, positions):
        pred = 1 if logit > 0 else 0
        pos_s1, pos_s2 = pos[0], pos[1]

        if pred == 0:
            g.addEdge(pos_s2, pos_s1)
        elif pred == 1:
            g.addEdge(pos_s1, pos_s2)

    while g.isCyclic():
        g.isCyclic()

    order = g.topologicalSort()
    gold_order = list(range(nvert))
    return calculate_metrics_list(order, gold_order)
