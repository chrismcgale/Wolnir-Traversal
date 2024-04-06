struct COOGraph {
    int numEdges;
    unsigned int* src;
    unsigned int* dst;
}

__host__ unsigned int* edge_bfs(COOGraph cooGraph, unsigned int startVertex) {
    COOGraph* cooGraph_d;
    unsigned int* newVertex_h, *newVertex_d;
    *newVertex_h = 1

    unsigned int level_h[cooGraph.numVertices], *level_d;

    memset(level_h, UINT_MAX, sizeof(unsigned int) * cooGraph.numVertices);

    // Can have multiple starts!!
    level_h[startVertex] = 0;

    cudaMalloc((void**)&cooGraph_d, sizeof(COOGraph));
    cudaMalloc((void**)&level_d, sizeof(level));
    cudaMalloc((void**)&newVertex_d, sizeof(unsigned int));

    cudaMemcpy(cooGraph_d, cooGraph, sizeof(COOGraph) s, cudaMemcpyHostToDevice);
    cudaMemcpy(level_d, level_h, sizeof(unsigned int) * cooGraph.numVertices, cudaMemcpyHostToDevice);
    cudaMemcpy(newVertex_d, newVertex_h, sizeof(unsigned int), cudaMemcpyHostToDevice);


    while (*newVertex_h == 1) {
        *newVertex_h = 0;
        cudaMemcpy(newVertex_d, newVertex_h, sizeof(unsigned int), cudaMemcpyHostToDevice);
        edge_bfs_kernel<<<(cooGraph.numVertices / 256), 256>>>(cooGraph, level, newVertex, currLevel);
        cudaMemcpy(newVertex_h, newVertex_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        currLevel++;
    }

    cudaMemcpy(level_h, level_d, sizeof(unsigned int) * cooGraph.numVertices, cudaMemcpyDeviceToHost);

    cudaFree(cooGraph);
    cudaFree(level_d);
    cudaFree(newVertex_d);

    return level_h;

}


__global__ void edge_bfs_kernel(COOGraph cooGraph, unsigned int* level, unsigned int* newVertex, unsigned int currLevel) {
    unsigned int edge = blockIdx.x*blockDim.x + thread.Idx.x;
    if (edge < cooGraph.numEdges) {
        unsigned int vertex = cooGraph.src[edge];
        if (level[vertex] == currLevel - 1) {
            unsigned int neighbour = cooGraph.dst[edge];
            if (level[neighbour] == UINT_MAX) { // Not yet visited
                level[neighbour] = currLevel;
                *newVertex = 1;
            }
        }
    }
}