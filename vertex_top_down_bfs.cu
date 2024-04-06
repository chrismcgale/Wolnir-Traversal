struct CSRGraph {
    int numVertices;
    unsigned int* srcPtrs;
    unsigned int* dst;
}


__host__ unsigned int* vertex_top_down_bfs(CSRGraph csrGraph, unsigned int startVertex) {
    unsigned int* newVertex;

    *newVertex = 1

    unsigned int level[csrGraph.numVertices];

    memset(level, UINT_MAX, sizeof(level));

    // Can have multiple starts!!
    level[startVertex] = 0;

    cudaMalloc((void**)csrGraph, sizeof(csrGraph));
    cudaMalloc((void**)level, sizeof(level));
    cudaMalloc((void**)&newVertex, sizeof(unsigned int));

    unsigned int currLevel = 1;
    unsigned int hostLevel = 1;
    while (hostLevel == 1) {
        hostLevel = 0;
        cudaMemcpy(&hostLevel, newVertex, cudaMemcpyHostToDevice);
        vertex_top_down_bfs_kernel<<<(csrGraph.numVertices / 256), 256>>>(csrGraph, level, newVertex, currLevel);
        cudaMemcpy(newVertex, &hostLevel, cudaMemcpyDeviceToHost);
        currLevel++;
    }

    cudaFree(csrGraph);
    cudaFree(level);
    cudaFree(newVertex);

}




__global__ void vertex_top_down_bfs_kernel(CSRGraph csrGraph, unsigned int* level, unsigned int* newVertex, unsigned int currLevel) {
    unsigned int vertex = blockIdx.x*blockDim.x + thread.Idx.x;
    if (vertex < csrGraph.numVertices) {
        if (level[vertex] == currLevel - 1) {
            for (unsigned int edge = csrGraph.srcPtrs[vertex]; edge < csrGraph.srcPtrs[vertex + 1]; edge++) {
                unsigned int neighbour = csrGraph.dst[edge];
                if (level[neighbour] == UINT_MAX) { // Not yet visited
                    level[neighbour] = currLevel;
                    *newVertex = 1;
                }
            }
        }
    }
}