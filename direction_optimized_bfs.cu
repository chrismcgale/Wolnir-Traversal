// Top down (push) is better for small number of vertices per level (earlier) while bottom up (pull) is better for larger number of vertices per level (latter)
// We can improve performance by switching from push to pull midway through our search

// Push needs CSR and Pull needs CSC so either both must be provided or the graph needs to guarantee undirected edges (In which case CSR and CSC are equivalent)
struct Graph {
    int numVertices;
    unsigned int* srcPtrs;
    unsigned int* dst;
}

// Switching level depends on the degree of the graph (low-degree = many levels = more levels needed to reach critical mass)
__host__ direction_optimized_bfs(Graph graph, unsigned int startVertex, unsigned int switchingLevel) {
    unsigned int* newVertex;

    *newVertex = 1

    unsigned int level[graph.numVertices];

    memset(level, UINT_MAX, sizeof(level));
    
    // Can have multiple starts!!
    level[startVertex] = 0;

    cudaMalloc((void**)graph, sizeof(graph));
    cudaMalloc((void**)level, sizeof(level));
    cudaMalloc((void**)&newVertex, sizeof(unsigned int));

    unsigned int currLevel = 1;
    unsigned int hostLevel = 1;
    while (hostLevel == 1) {
        hostLevel = 0;
        cudaMemcpy(&hostLevel, newVertex, cudaMemcpyHostToDevice);

        if (currLevel < switchingLevel) {
            vertex_top_down_bfs_kernel<<<(graph.numVertices / 256), 256>>>(graph, level, newVertex, currLevel);
        } else {
            vertex_bottom_up_bfs_kernel<<<(cscGraph.numVertices / 256), 256>>>(cscGraph, level, newVertex, currLevel);
        }

        cudaMemcpy(newVertex, &hostLevel, cudaMemcpyDeviceToHost);
        currLevel++;
    }

    cudaFree(graph);
    cudaFree(level);
    cudaFree(newVertex);

}
