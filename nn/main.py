"""Let's get that max clique"""
from nn.read_dimacs import read_dimacs
from nn.trainers import train_test
import time
import os


def main(file):
    """
    Read a file pass it into the graph network, get your probabilities, use them to find max clique
    """
    (vert, edges, neighs) = read_dimacs(file, 1)
    grade = [0] * len(vert)
    for node in range(len(vert)):  # bereken de graad van iedere node
        grade[node] = len(neighs[node])

    for j in range(len(vert)):
        # voeg loops toe van de nodes naar zichzelf zodat hun originele waarde iets zwaarder weegt
        edges.append((j, j))
        neighs[j].append(j)

    start_time = time.time()

    probs = train_test(0.5, edges, vert, grade)

    vals = []
    for (j, prob) in enumerate(probs):  # uiteindelijke waarden:
        val = probs[j].item()
        count = len(neighs[j])
        for n in neighs[j]:
            val += probs[n].item()*grade[n]
        vals.append(
            val/count)  # normalizeer met aantal buren want mag niet van invloed zijn hier
    probs = vals

    # we kiezen de grootste top en eventueel toppen die dicht bij max liggen als kandidaten voor max kliek
    max_prob = max(probs)
    
    best_cliq = []

    candidates = [i for i, prob in enumerate(probs) if prob >= max_prob - 0.1]

    print("candidates: ", len(candidates))
    for best_node in candidates:  # we proberen iteratief de huidige kliek uit te breiden
        max_clique = [best_node]
        broken = False
        while not broken:
            updated = []
            for node in max_clique:  # zoek alle buren van de huidige kliek
                for e in neighs[node]:
                    if e not in updated:
                        updated.append(e)

            # probeer eerst degene met meer kans
            updated.sort(key=lambda x: probs[x], reverse=True)
            test = max_clique.copy()

            if not updated:
                broken = True

            for upd in updated:  # alle buren die de kliek niet zouden breken gaan we toevoegen
                test.append(upd)
                if is_clique(test, neighs):
                    max_clique.append(upd)
                else:
                    test.pop()
                    broken = True  # als er een buur was die niet tot kliek hoorde stoppen we na deze iteratie
        if (len(max_clique) > len(best_cliq)):
            best_cliq = max_clique

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")


    print("Best found clique: ", best_cliq)
    print("Max found kliekgetal: ", len(best_cliq))


def is_clique(nodes, neighs):
    """
    Check if the list of nodes forms a clique based on their neighbors
    """
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if nodes[j] not in neighs[nodes[i]]:
                return False
    return True


if __name__ == "__main__":
    pad = "C:\\Users\\Jibril\\Desktop\\disalgo-taak1\\testfiles\\"
    for file_pad in os.listdir(pad):
        if file_pad.endswith(".txt"):
            print(pad + file_pad)
            main(pad + file_pad)