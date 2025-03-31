#include "engine.h"


// IMPORTANT: Best practice is to assign most used models the lowest IDs possible
__device__ Mesh* loadedModels = nullptr; // pointer to loaded models in global memory
__device__ int loadedModelCount = 0; // number of loaded models in global memory




__device__ void saveMesh(Mesh mesh) {
    // Save the mesh to global memory
    loadedModels[loadedModelCount] = mesh;
    loadedModelCount++;
}

__device__ void freeMesh(int modelID) {
    // Find the mesh with the matching ID and remove it
    for (int i = 0; i < loadedModelCount; i++) {
        if (loadedModels[i].modelID == modelID) {
            // Shift remaining meshes down to fill the gap
            for (int j = i; j < loadedModelCount - 1; j++) {
                loadedModels[j] = loadedModels[j + 1];
            }
            loadedModelCount--;
            break;
        }
    }
}

interpolator tickLogic(int tickCount) {
    interpolator result;
    result.tickCount = tickCount;
    

    return result;
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