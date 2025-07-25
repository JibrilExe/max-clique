"""Write graph files dimacs style"""

def write_dimacs(filename, num_vertices, edges):
    """
    Write a graph to a file DIMACS format

    keyword arguments:
    filename -- output file
    num_vertices -- amount of nodes in graph
    edges -- list of lists or tuples describing the edges in the graph
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"c number of vertices  : {num_vertices}\n")
        for edge in edges:
            f.write(f"e {edge[0]} {edge[1]}\n")
