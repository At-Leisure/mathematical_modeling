import numpy as np


def ant_colony_algorithm(distance_matrix, num_ants=100, num_iterations=100, pheromone_evaporation=0.5, alpha=1,
                            beta=1):
    """ 蚁群算法2.0
    ## Return
    `best_distance`-距离
    `best_path`-顺序"""
    num_cities = len(distance_matrix)

    # 初始化信息素矩阵
    pheromone_matrix = np.ones((num_cities, num_cities)) / num_cities

    # 记录最佳路径和距离
    best_path = None
    best_distance = float('inf')

    for iteration in range(num_iterations):
        # 初始化每只蚂蚁的路径和距离
        ant_paths = []
        ant_distances = []

        for ant in range(num_ants):
            # 初始化蚂蚁当前所在的城市
            current_city = np.random.randint(num_cities)
            visited_cities = [current_city]
            path = [current_city]
            distance = 0

            # 构建完整路径
            while len(visited_cities) < num_cities:
                # 计算转移概率
                probabilities = []
                for city in range(num_cities):
                    if city not in visited_cities:
                        pheromone = pheromone_matrix[current_city][city]
                        heuristic = 1 / \
                            distance_matrix[current_city][city] ** beta
                        probabilities.append(
                            (city, pheromone ** alpha * heuristic))

                # 根据转移概率选择下一个城市
                probabilities = np.array(probabilities)
                probabilities[:, 1] /= np.sum(probabilities[:, 1])
                next_city = np.random.choice(
                    probabilities[:, 0], p=probabilities[:, 1])

                # 更新路径和距离
                path.append(int(next_city))
                visited_cities.append(int(next_city))
                distance += distance_matrix[current_city][int(next_city)]
                current_city = int(next_city)

            # 回到起始城市
            path.append(path[0])
            distance += distance_matrix[current_city][path[0]]

            ant_paths.append(path)
            ant_distances.append(distance)

            # 更新最佳路径和距离
            if distance < best_distance:
                best_path = path
                best_distance = distance

        # 更新信息素矩阵
        delta_pheromone_matrix = np.zeros((num_cities, num_cities))
        for ant in range(num_ants):
            for i in range(len(ant_paths[ant]) - 1):
                current_city = ant_paths[ant][i]
                next_city = ant_paths[ant][i + 1]
                delta_pheromone_matrix[current_city][next_city] += 1 / \
                    ant_distances[ant]

        pheromone_matrix = (1 - pheromone_evaporation) * \
            pheromone_matrix + delta_pheromone_matrix

    return best_distance, best_path