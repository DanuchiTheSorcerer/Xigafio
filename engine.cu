#include "engine.h"
#include <mma.h>
#include <cuda_fp16.h>


using namespace nvcuda;

interpolator tickLogic(int tickCount) {
    interpolator result;
    result.tickCount = tickCount;
    result.freedModelCount = 0;
    result.newModelCount = 0;

    return result;
}





// IMPORTANT: Best practice is to assign most used models the lowest IDs possible
__device__ Model* loadedModels = nullptr; // pointer to loaded models in global memory
__device__ int loadedModelCount = 0; // number of loaded models in global memory


__device__ void saveModel(Model model) {
    // Save the mesh to global memory
    loadedModels[loadedModelCount] = model;
    loadedModelCount++;
}

__device__ void freeModel(int modelID) {
    // Mark the mesh for removal by setting its modelID to -1
    for (int i = 0; i < loadedModelCount; i++) {
        if (loadedModels[i].modelID == modelID) {
            loadedModels[i].modelID = -1;
        }
    }

    // Compact the array to remove marked meshes
    int writeIdx = 0;
    for (int i = 0; i < loadedModelCount; i++) {
        if (loadedModels[i].modelID != -1) {
            loadedModels[writeIdx++] = loadedModels[i];
        }
    }
    loadedModelCount = writeIdx;
}


struct SceneObject {
    Triangle* triangles;
    int triangleCount;
    bool isStatic;
};

__device__ SceneObject* worldScene = nullptr; // pointer to objects in global memory
__device__ int worldSceneCount = 0; // number of objects in world space




__global__ void computePixel(uint32_t* buffer, int width, int height, const interpolator* interp,float inpf) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        int modVal = interp->tickCount;
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


__global__ void meshesToWorld(Mesh* meshes, Model* loadedModels, int loadedModelCount, bool dynamic) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    if (meshes[bid].dynamic ^ dynamic) {
        return; 
    }
    // Load transformation parameters into shared memory
    __shared__ float positionX, positionY, positionZ;
    __shared__ float rotationX, rotationY, rotationZ, rotationAngle;
    __shared__ float scale;
    __shared__ Mesh *mesh = &meshes[bid];
    __shared__ int modelID;
    __shared__ half rodriguesMatrix[4][4];



    positionX = mesh->position[0];
    positionY = mesh->position[1];
    positionZ = mesh->position[2];
    rotationX = mesh->rotation[0];
    rotationY = mesh->rotation[1];
    rotationZ = mesh->rotation[2];
    rotationAngle = mesh->rotation[3];
    scale = mesh->scale;
    modelID = mesh->modelID;

    SceneObject output;
    output.triangleCount = loadedModels[modelID].triangleCount;
    output.isStatic = dynamic;



    __syncthreads(); // Ensure shared memory is populated


    __shared__ float norm, rX, rY, rZ, cosTheta, sinTheta, oneMinusCos;

    // Normalize rotation axis
    if (tid == 0) {
        float norm = sqrtf(rotationX * rotationX + rotationY * rotationY + rotationZ * rotationZ);
        if (norm > 0.0f) {
            rX = rotationX / norm;
            rY = rotationY / norm;
            rZ = rotationZ / norm;
        }
    }
    if (tid == 1) {
        // Compute sine and cosine terms
        cosTheta = cosf(rotationAngle);
        oneMinusCos = 1.0f - cosTheta;
    }
    if (tid == 2) {
        sinTheta = sinf(rotationAngle);
    }

    __syncthreads(); // Ensure shared memory is populated

    // Compute the Rodrigues rotation matrix
    // Compute shared intermediate values in parallel
    __shared__ float sx, sy, sz, tXX, tYY, tZZ, tXY, tXZ, tYZ;

    if (tid == 0) sx = sinTheta * rX;
    if (tid == 1) sy = sinTheta * rY;
    if (tid == 2) sz = sinTheta * rZ;
    if (tid == 3) tXX = oneMinusCos * rX * rX;
    if (tid == 4) tYY = oneMinusCos * rY * rY;
    if (tid == 5) tZZ = oneMinusCos * rZ * rZ;
    if (tid == 6) tXY = oneMinusCos * rX * rY;
    if (tid == 7) tXZ = oneMinusCos * rX * rZ;
    if (tid == 8) tYZ = oneMinusCos * rY * rZ;

    __syncthreads(); // Ensure precomputed values are available

    // Compute scaled rotation matrix elements in parallel
    if (tid == 0) rodriguesMatrix[0][0] = __float2half(scale * (tXX + cosTheta));
    if (tid == 1) rodriguesMatrix[0][1] = __float2half(scale * (tXY - sz));
    if (tid == 2) rodriguesMatrix[0][2] = __float2half(scale * (tXZ + sy));
    if (tid == 3) rodriguesMatrix[0][3] = __float2half(positionX);

    if (tid == 4) rodriguesMatrix[1][0] = __float2half(scale * (tXY + sz));
    if (tid == 5) rodriguesMatrix[1][1] = __float2half(scale * (tYY + cosTheta));
    if (tid == 6) rodriguesMatrix[1][2] = __float2half(scale * (tYZ - sx));
    if (tid == 7) rodriguesMatrix[1][3] = __float2half(positionY);

    if (tid == 8) rodriguesMatrix[2][0] = __float2half(scale * (tXZ - sy));
    if (tid == 9) rodriguesMatrix[2][1] = __float2half(scale * (tYZ + sx));
    if (tid == 10) rodriguesMatrix[2][2] = __float2half(scale * (tZZ + cosTheta));
    if (tid == 11) rodriguesMatrix[2][3] = __float2half(positionZ);

    if (tid == 12) {
        rodriguesMatrix[3][0] = __float2half(0.0f);
        rodriguesMatrix[3][1] = __float2half(0.0f);
        rodriguesMatrix[3][2] = __float2half(0.0f);
        rodriguesMatrix[3][3] = __float2half(1.0f);
    }

    __syncthreads();
}


__device__ void interpolatorUpdateHandler(interpolator* interp) {
    // save needed models to global memory

    for (int i = 0; i < interp->newModelCount; i++) {
        Model model = interp->newModels[i];
        saveModel(model);
    }

    // remove unneeded models from global memory 
    for (int i = 0; i < interp->freedModelCount; i++) {
        int modelID = interp->freedModelIDs[i];
        freeModel(modelID);
    }

    // load static objects into scene
}