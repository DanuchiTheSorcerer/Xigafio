#include "engine.h"


interpolator tickLogic(int tickCount) {
    interpolator result;
    result.tickCount = tickCount;
    

    return result;
}





// IMPORTANT: Best practice is to assign most used models the lowest IDs possible
__device__ Mesh* loadedModels = nullptr; // pointer to loaded models in global memory
__device__ int loadedModelCount = 0; // number of loaded models in global memory


__device__ void saveMesh(Mesh mesh) {
    // Save the mesh to global memory
    loadedModels[loadedModelCount] = mesh;
    loadedModelCount++;
}

__global__ void freeMeshKernel(Mesh* models, int* modelCount, int modelID) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < *modelCount && models[idx].modelID == modelID) {
        // Mark the mesh for removal by setting its modelID to -1
        models[idx].modelID = -1;
    }
}

__device__ void freeMesh(int modelID) {
    // Launch a kernel to mark the mesh for removal
    int threadsPerBlock = 256;
    int blocksPerGrid = (loadedModelCount + threadsPerBlock - 1) / threadsPerBlock;
    freeMeshKernel<<<blocksPerGrid, threadsPerBlock>>>(loadedModels, &loadedModelCount, modelID);
    cudaDeviceSynchronize();

    // Compact the array to remove marked meshes
    int writeIdx = 0;
    for (int i = 0; i < loadedModelCount; i++) {
        if (loadedModels[i].modelID != -1) {
            loadedModels[writeIdx++] = loadedModels[i];
        }
    }
    loadedModelCount = writeIdx;
}





__global__ void computePixel(uint32_t* buffer, int width, int height, const interpolator* interp,float inpf) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        int modVal = interp->tickCount * 10;
        uint32_t red   = 128 + 127*sinf((float)x/128 + ((float)modVal*inpf)/60);
        uint32_t green = 128 + 127*cosf((float)y/128 + ((float)modVal*inpf)/60);
        uint32_t blue  = 128 + 12*sinf((float)x/128 + (float)y/128 + ((float)modVal*inpf)/6);
        buffer[idx] = 0xFF000000 | (red << 16) | (green << 8) | blue;
    }
}


__device__ void computeFrame(uint32_t* buffer, int width, int height, const interpolator* interp,float interpolationFactor) {
    // Define thread block and grid dimensions for the child kernel.
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Launch the child kernel to compute pixels.
    computePixel<<<numBlocks, threadsPerBlock>>>(buffer, width, height, interp,interpolationFactor);
    
    // Wait for the child kernel to finish before completing.
    __threadfence();
}

__device__ void interpolatorUpdateHandler(interpolator* interp) {
    // save needed meshes to global memory

    for (int i = 0; i < interp->newMeshCount; i++) {
        Mesh mesh = interp->newMeshes[i];
        saveMesh(mesh);
    }

    // remove unneeded meshes from global memory 
    for (int i = 0; i < interp->freedMeshCount; i++) {
        int modelID = interp->freedMeshIDs[i];
        freeMesh(modelID);
    }
}