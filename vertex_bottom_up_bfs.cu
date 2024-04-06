struct CSCGraph {
    int numVertices;
    unsigned int* dstPtrs;
    unsigned int* src;
}


__host__ unsigned int* vertex_bottom_up_bfs(CSCGraph cscGraph, unsigned int startVertex) {
    unsigned int* newVertex;

    *newVertex = 1

    unsigned int level[cscGraph.numVertices];

    memset(level, UINT_MAX, sizeof(level));

    // Can have multiple starts!!
    level[startVertex] = 0;

    cudaMalloc((void**)cscGraph, sizeof(cscGraph));
    cudaMalloc((void**)level, sizeof(level));
    cudaMalloc((void**)&newVertex, sizeof(unsigned int));

    unsigned int currLevel = 1;
    unsigned int hostLevel = 1;
    while (hostLevel == 1) {
        hostLevel = 0;
        cudaMemcpy(&hostLevel, newVertex, cudaMemcpyHostToDevice);
        vertex_bottom_up_bfs_kernel<<<(cscGraph.numVertices / 256), 256>>>(cscGraph, level, newVertex, currLevel);
        cudaMemcpy(newVertex, &hostLevel, cudaMemcpyDeviceToHost);
        currLevel++;
    }

    cudaFree(cscGraph);
    cudaFree(level);
    cudaFree(newVertex);

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