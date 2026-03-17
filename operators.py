def rosenbrock(x_list):
    """Funkcja Rosenbrocka dla wielu zmiennych N"""
    total = 0
    for i in range(len(x_list) - 1):
        total += 100 * (x_list[i+1] - x_list[i]**2)**2 + (x_list[i] - 1)**2
    return total

def single_point_crossover(parent1, parent2):
    """Krzyżowanie jednopunktowe [cite: 10, 16]"""
    point = np.random.randint(1, len(parent1) - 1)
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2