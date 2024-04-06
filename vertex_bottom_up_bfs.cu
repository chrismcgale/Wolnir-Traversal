struct CSCGraph {
    int numVertices;
    unsigned int* dstPtrs;
    unsigned int* src;
}


__host__ unsigned int* vertex_bottom_up_bfs(CSCGraph cscGraph, unsigned int startVertex) {
    CSCGraph* csrGraph_d;
    unsigned int* newVertex_h, *newVertex_d;
    *newVertex_h = 1

    unsigned int level_h[cscGraph.numVertices], *level_d;

    memset(level_h, UINT_MAX, sizeof(unsigned int) * cscGraph.numVertices);

    // Can have multiple starts!!
    level_h[startVertex] = 0;

    cudaMalloc((void**)&csrGraph_d, sizeof(CSCGraph));
    cudaMalloc((void**)&level_d, sizeof(level));
    cudaMalloc((void**)&newVertex_d, sizeof(unsigned int));

    cudaMemcpy(csrGraph_d, cscGraph, sizeof(CSCGraph) s, cudaMemcpyHostToDevice);
    cudaMemcpy(level_d, level_h, sizeof(unsigned int) * cscGraph.numVertices, cudaMemcpyHostToDevice);
    cudaMemcpy(newVertex_d, newVertex_h, sizeof(unsigned int), cudaMemcpyHostToDevice);


    while (*newVertex_h == 1) {
        *newVertex_h = 0;
        cudaMemcpy(newVertex_d, newVertex_h, sizeof(unsigned int), cudaMemcpyHostToDevice);
        vertex_top_down_bfs_kernel<<<(cscGraph.numVertices / 256), 256>>>(cscGraph, level, newVertex, currLevel);
        cudaMemcpy(newVertex_h, newVertex_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        currLevel++;
    }

    cudaMemcpy(level_h, level_d, sizeof(unsigned int) * cscGraph.numVertices, cudaMemcpyDeviceToHost);

    cudaFree(cscGraph);
    cudaFree(level_d);
    cudaFree(newVertex_d);

    return level_h;

}






__global__ void vertex_bottom_up_bfs_kernel(CSCGraph cscGraph, unsigned int* level, unsigned int* newVertex, unsigned int currLevel) {
    unsigned int vertex = blockIdx.x*blockDim.x + thread.Idx.x;
    if (vertex < cscGraph.numVertices) {
        if (level[vertex] == UINT_MAX) {  // Not yet visited
            for (unsigned int edge = cscGraph.dstPtrs[vertex]; edge < cscGraph.dstPtrs[vertex + 1]; edge++) {
                unsigned int neighbour = cscGraph.src[edge];
                if (level[neighbour] == currLevel - 1) {
                    level[vertex] = currLevel;
                    *newVertex = 1;
                    // Can be much better than top_down bc less control divergence
                    break;
                }
            }
        }
    }
}