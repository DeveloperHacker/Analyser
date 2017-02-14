

def KMeans(clusters: list, num_clusters=None):
    if num_clusters is None:
        num_clusters = len(clusters)
    clusters = list(reversed(sorted(clusters, key=lambda cluster: len(cluster))))
    for i in range(0, round(num_clusters)):
        print(clusters[i])


def XNeighbors(clusters: list, num_clusters=None):
    if num_clusters is None:
        num_clusters = len(clusters)
    # clusters = list(reversed(sorted(clusters, key=lambda target: len(target["data"]))))
    clusters.sort(key=lambda target: len(target.data), reverse=True)
    for i in range(0, round(num_clusters)):
        print("\"%s\":" % clusters[i].label)
        for label, distance in clusters[i].data:
            print("\t%.3f: \"%s\"" % (distance, label))
