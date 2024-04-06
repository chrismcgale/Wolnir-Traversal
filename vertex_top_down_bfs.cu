struct CSRGraph {
    int numVertices;
    unsigned int* srcPtrs;
    unsigned int* dst;
}


__host__ unsigned int* vertex_top_down_bfs(CSRGraph csrGraph, unsigned int startVertex) {
    CSRGraph* csrGraph_d;
    unsigned int* newVertex_h, *newVertex_d;
    *newVertex_h = 1

    unsigned int level_h[csrGraph.numVertices], *level_d;

    memset(level_h, UINT_MAX, sizeof(unsigned int) * csrGraph.numVertices);

    // Can have multiple starts!!
    level_h[startVertex] = 0;

    cudaMalloc((void**)&csrGraph_d, sizeof(CSRGraph));
    cudaMalloc((void**)&level_d, sizeof(level));
    cudaMalloc((void**)&newVertex_d, sizeof(unsigned int));

    cudaMemcpy(csrGraph_d, csrGraph, sizeof(CSRGraph) s, cudaMemcpyHostToDevice);
    cudaMemcpy(level_d, level_h, sizeof(unsigned int) * csrGraph.numVertices, cudaMemcpyHostToDevice);
    cudaMemcpy(newVertex_d, newVertex_h, sizeof(unsigned int), cudaMemcpyHostToDevice);


    while (*newVertex_h == 1) {
        *newVertex_h = 0;
        cudaMemcpy(newVertex_d, newVertex_h, sizeof(unsigned int), cudaMemcpyHostToDevice);
        vertex_top_down_bfs_kernel<<<(csrGraph.numVertices / 256), 256>>>(csrGraph, level, newVertex, currLevel);
        cudaMemcpy(newVertex_h, newVertex_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        currLevel++;
    }

    cudaMemcpy(level_h, level_d, sizeof(unsigned int) * csrGraph.numVertices, cudaMemcpyDeviceToHost);

    cudaFree(csrGraph);
    cudaFree(level_d);
    cudaFree(newVertex_d);

    return level_h;

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