def printClusters(clusters: list, numClusters: int):
    clusters = list(reversed(sorted(clusters, key=lambda x: len(x))))
    for i in range(0, numClusters * 0.05):
        print(clusters[i])
