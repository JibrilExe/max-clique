"""
Reader for DIMACS file to internal representation (V,E).
"""
def read_dimacs(path, start_nr):
    """
    Read in file in DIMACS format and return adjacency matrix and edge list.
    """
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    num_vertices = 0
    for line in lines:
        if line.startswith('p'): # example: p col 125 6963
            parts = line.split()
            num_vertices = int(parts[2])
            break
        elif line.startswith('c number of vertices'):
            parts = line.split(':')
            num_vertices = int(parts[1].strip())
            break

    neighs = [[] for _ in range(num_vertices)]
    edges = []
    vertices = [[0] for _ in range(num_vertices)]
    for line in lines:
        if line.startswith('e'):
            parts = line.split()
            u = int(parts[1]) - start_nr
            v = int(parts[2]) - start_nr
            neighs[u].append(v)
            neighs[v].append(u)
            edges.append((u, v))

    return (vertices, edges, neighs)

if __name__ == '__main__':
    read_dimacs('C125.9.clq', 1)
