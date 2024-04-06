struct CSRGraph {
    int numVertices;
    unsigned int* srcPtrs;
    unsigned int* dst;
}


__host__ unsigned int* frontier_bfs(CSRGraph csrGraph, unsigned int startVertex, unsigned int max_degree) {
    CSRGraph* csrGraph_d;
    unsigned int* numCurrFrontier_h, *numCurrFrontier_d;
    *numCurrFrontier_h = 1

    unsigned int level_h[csrGraph.numVertices], *level_d;

    memset(level_h, UINT_MAX, sizeof(unsigned int) * csrGraph.numVertices);

    // Can have multiple starts!!
    level_h[startVertex] = 0;

    cudaMalloc((void**)&csrGraph_d, sizeof(CSRGraph));
    cudaMalloc((void**)&level_d, sizeof(level));
    cudaMalloc((void**)&numCurrFrontier_d, sizeof(unsigned int));

    cudaMemcpy(csrGraph_d, csrGraph, sizeof(CSRGraph) s, cudaMemcpyHostToDevice);
    cudaMemcpy(level_d, level_h, sizeof(unsigned int) * csrGraph.numVertices, cudaMemcpyHostToDevice);
    cudaMemcpy(numCurrFrontier_d, numCurrFrontier_h, sizeof(unsigned int), cudaMemcpyHostToDevice);


    unsigned int numPrevFrontier = 1;
    unsigned int* prevFrontier_h, *prevFrontier_d;
    cudaMalloc((void**)&prevFrontier_d, sizeof(unsigned int) * max_degree);

    while (numPrevFrontier > 0) {
        cudaMemcpy(prevFrontier_d, prevFrontier_h, sizeof(unsigned int), cudaMemcpyHostToDevice);

        unsigned int* currFrontier; 
        cudaMalloc((void**)&currFrontier, sizeof(unsigned int) * max_degree);

        *numCurrFrontier_h = 0;
        cudaMemcpy(numCurrFrontier_d, numCurrFrontier_h, sizeof(unsigned int), cudaMemcpyHostToDevice);

        frontier_bfs_kernel<<<(csrGraph.numVertices / 256), 256>>>(csrGraph, level, newVertex, prevFrontier_d, numPrevFrontier, currFrontier, numCurrFrontier_d, currLevel);

        cudaMemcpy(prevFrontier_h, currFrontier, sizeof(unsigned int) * max_degree, cudaMemcpyDeviceToHost);
        cudaMemcpy(numPrevFrontier, numCurrFrontier_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        cudaFree(currFrontier);
        currLevel++;
    }

    cudaMemcpy(level_h, level_d, sizeof(unsigned int) * csrGraph.numVertices, cudaMemcpyDeviceToHost);

    cudaFree(csrGraph);
    cudaFree(level_d);
    cudaFree(numCurrFrontier_d);
    cudaFree(prevFrontier);

    return level_h;

}




__global__ void frontier_bfs_kernel(CSRGraph csrGraph, unsigned int* level, 
                unsigned int* prevFrontier, unsigned int numPrevFrontier, 
                unsigned int* currFrontier. unsigned int* numCurrFrontier, unsigned int currLevel) {
    unsigned int i = blockIdx.x*blockDim.x + thread.Idx.x;
    if (i < numPrevFrontier) {
        unsigned int vertex = prevFrontier[i];
        for (unsigned int edge = csrGraph.srcPtrs[vertex]; edge < csrGraph.srcPtrs[vertex + 1]; edge++) {
            unsigned int neighbour = csrGraph.dst[edge];
            if (atomicCAS(&level[neighbour], UINT_MAX, currLevel) == UINT_MAX) { // Not yet visited
                unsigned int currFrontierIdx = atomicAdd(numCurrFrontier, 1);
                currFrontier[currFrontierIdx] = neighbour;
            }
        }
    }
}