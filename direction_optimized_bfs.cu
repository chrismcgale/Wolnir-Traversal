// Top down (push) is better for small number of vertices per level (earlier) while bottom up (pull) is better for larger number of vertices per level (latter)
// We can improve performance by switching from push to pull midway through our search

// Push needs CSR and Pull needs CSC so either both must be provided or the graph needs to guarantee undirected edges (In which case CSR and CSC are equivalent)
struct Graph {
    int numVertices;
    unsigned int* srcPtrs;
    unsigned int* dst;
}

// Switching level depends on the degree of the graph (low-degree = many levels = more levels needed to reach critical mass)
__host__ unsigned int* edge_bfs(Graph graph, unsigned int startVertex) {
    Graph* cooGraph_d;
    unsigned int* newVertex_h, *newVertex_d;
    *newVertex_h = 1

    unsigned int level_h[graph.numVertices], *level_d;

    memset(level_h, UINT_MAX, sizeof(unsigned int) * graph.numVertices);

    // Can have multiple starts!!
    level_h[startVertex] = 0;

    cudaMalloc((void**)&cooGraph_d, sizeof(Graph));
    cudaMalloc((void**)&level_d, sizeof(level));
    cudaMalloc((void**)&newVertex_d, sizeof(unsigned int));

    cudaMemcpy(cooGraph_d, graph, sizeof(Graph) s, cudaMemcpyHostToDevice);
    cudaMemcpy(level_d, level_h, sizeof(unsigned int) * graph.numVertices, cudaMemcpyHostToDevice);
    cudaMemcpy(newVertex_d, newVertex_h, sizeof(unsigned int), cudaMemcpyHostToDevice);


    while (*newVertex_h == 1) {
        *newVertex_h = 0;
        cudaMemcpy(newVertex_d, newVertex_h, sizeof(unsigned int), cudaMemcpyHostToDevice);
        if (currLevel < switchingLevel) {
            vertex_top_down_bfs_kernel<<<(graph.numVertices / 256), 256>>>(graph, level, newVertex, currLevel);
        } else {
            vertex_bottom_up_bfs_kernel<<<(cscGraph.numVertices / 256), 256>>>(cscGraph, level, newVertex, currLevel);
        }
        cudaMemcpy(newVertex_h, newVertex_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        currLevel++;
    }

    cudaMemcpy(level_h, level_d, sizeof(unsigned int) * graph.numVertices, cudaMemcpyDeviceToHost);

    cudaFree(graph);
    cudaFree(level_d);
    cudaFree(newVertex_d);

    return level_h;

}
