"""Try to find best params"""

import optuna
from nn.trainers import train_test
from nn.read_dimacs import read_dimacs

y_true = [
    [0,1,2,3],
    [0,1,4,5],
    [0,2,4,5,1],
    [1,2,5,7,8],
    [0,1,5,13],
    [0,1,2,3],
    [7,8,9],
    [9,11,12,10,13],
    [9,11,12,10,13]
]

verts = []
edgess = []
gradess = []
base = 'C:\\Users\\Jibril\\Desktop\\disalgo-taak1\\testfiles\\'
for i in range(1,5):
    print(i)
    path = base + 'test' + str(i) + '.dimacs'
    (vert, edges) = read_dimacs(path)
    verts.append(vert)
    edgess.append(edges)
    grade = [0] * len(vert)
    for edge in edges:
        for node in edge:
            grade[node] += 1
    gradess.append(grade)

def objective(trial):
    """
    Objective functie voor optuna,
    probeert beste params voor ons model te vinden
    idee is parameter zo kiezen zodat voor al onze kleine test grafen slaagde
    maar was niet zo simpel
    """
    b = trial.suggest_float('b', 0.1, 10.0, step=0.1)

    correct = 0.0
    false = 0.0
    
    with open('probs_output.txt', 'a') as f:
        f.write(f"Trial number: {trial.number}\n")
        for i in range(4):
            probs = train_test(b, edgess[i], verts[i], gradess[i])
            f.write(f"Iteration {i+1}, probs: {probs}\n")
            summer= 0.0
            false_summer = 0.0 # houd een som bij van goede en slechte predicties van het model
            for (j, prob) in enumerate(probs):
                if j in y_true[i]:
                    summer += prob
                else:
                    false_summer += prob
            summer /= len(y_true[i])
            false_summer /= (len(probs) - len(y_true[i]))
            correct += summer
            false += false_summer

    return correct - false #dit wordt gemaximaliseerd dus slechte predicties gaan naar min en goede naar max

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
