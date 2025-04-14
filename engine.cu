#include "engine.h"
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <windows.h>
#include "dynamicArray.h"


using namespace nvcuda;

constexpr float pi = 3.14159265358979323846f; // Define pi as a constant

interpolator tickLogic(int tickCount) {
    Model model ;
    model.modelID = 0;
    model.triangleCount = 1;
    model.triangles = new Triangle[model.triangleCount];
    model.triangles[0].points[0].x = 0.0f;
    model.triangles[0].points[0].y = 0.0f;
    model.triangles[0].points[0].z = 0.0f;
    model.triangles[0].points[1].x = 1.0f;
    model.triangles[0].points[1].y = 0.0f;
    model.triangles[0].points[1].z = 0.0f;
    model.triangles[0].points[2].x = 1.0f;
    model.triangles[0].points[2].y = 1.0f;
    model.triangles[0].points[2].z = 0.0f;


    Mesh testMesh;
    testMesh.modelID = 0;
    testMesh.scale = 1.0f;
    testMesh.position[0] = 0.0f;
    testMesh.position[1] = 0.0f;
    testMesh.position[2] = 0.0f;
    testMesh.rotation[0] = 0.0f;
    testMesh.rotation[1] = 1.0f;
    testMesh.rotation[2] = 0.0f;
    testMesh.rotation[3] = pi/2;
    testMesh.dynamic = false;





    interpolator result;
    result.tickCount = tickCount;
    result.freedModelCount = 0;
    result.newModelCount = 0;
    if (tickCount == 0) {
        //result.newModelCount = 1;
        //result.newModels[0] = model;
        //result.staticMeshCount = 1;
        //result.dynamicMeshCount = 0;
        //result.meshes[0] = testMesh;
    }


    return result;
    // for refrence
    // struct interpolator  {
    //     int tickCount;
    //     Mesh* meshes;
    //     int staticMeshCount;
    //     int dynamicMeshCount;
    //     Camera camera;
    //     Model* newModels;
    //     int newModelCount;
    //     int* freedModelIDs;
    //     int freedModelCount;
    // };
}





// IMPORTANT: Best practice is to assign most used models the lowest IDs possible


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
        int speed = 3;
        int modVal = interp->tickCount*speed;
        uint32_t red   = 128 + 127*sinf((float)x/128 + ((float)modVal+inpf*speed)/60);
        uint32_t green = 128 + 127*cosf((float)y/128 + ((float)modVal+inpf*speed)/60);
        uint32_t blue  = 128 + 12*sinf((float)x/128 + (float)y/128 + ((float)modVal+inpf*speed)/6);
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


__global__ void meshesToWorld(Mesh* meshes, Model* loadedModels, bool dynamic, SceneObject* output) { // takes in meshes and outputs to worldScene
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    if (meshes[bid].dynamic ^ dynamic) {
        return; 
    }
    // Load transformation parameters into shared memory
    __shared__ float positionX, positionY, positionZ;
    __shared__ float rotationX, rotationY, rotationZ, rotationAngle;
    __shared__ float scale;
    __shared__ int modelID;
    __shared__ half rodriguesMatrix[4][4];



    positionX = meshes[bid].position[0];
    positionY = meshes[bid].position[1];
    positionZ = meshes[bid].position[2];
    rotationX = meshes[bid].rotation[0];
    rotationY = meshes[bid].rotation[1];
    rotationZ = meshes[bid].rotation[2];
    rotationAngle = meshes[bid].rotation[3];
    scale = meshes[bid].scale;
    modelID = meshes[bid].modelID;

    int triangleCount = loadedModels[modelID].triangleCount;



    __syncthreads(); // Ensure shared memory is populated


    __shared__ float rX, rY, rZ, cosTheta, sinTheta, oneMinusCos;

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

    __shared__ half chunckedMatrix[16][16];
    if (tid < 16) {
        chunckedMatrix[tid / 4][tid % 4] = rodriguesMatrix[tid / 4][tid % 4];
        chunckedMatrix[4 + tid / 4][4 + tid % 4] = rodriguesMatrix[tid / 4][tid % 4];
        chunckedMatrix[8 + tid / 4][8 + tid % 4] = rodriguesMatrix[tid / 4][tid % 4];
        chunckedMatrix[12 + tid / 4][12 + tid % 4] = rodriguesMatrix[tid / 4][tid % 4];
    }

    __syncthreads();


    int batchNumber = (triangleCount + 19) / 20; // round up

    for (int i = 0; i < batchNumber;i++) {
        //batch points together
        __shared__ half batch[16][16];
        int threadSpotRow = tid/5;
        int threadSpotCol = tid%5;

        if (triangleCount % 20 == 0 || i* 20 + threadSpotRow * 5 + threadSpotCol < triangleCount) { // only let a thread batch if it has a triangle to batch
            for (int j = 0; j < 3;j++) {
                batch[threadSpotRow * 4][threadSpotCol * 3 + j] = __float2half(loadedModels[modelID].triangles[i * 20 + (threadSpotRow * 5 + threadSpotCol)].points[j].x);
                batch[1 + threadSpotRow * 4][threadSpotCol * 3 + j] = __float2half(loadedModels[modelID].triangles[i * 20 + (threadSpotRow * 5 + threadSpotCol)].points[j].y);
                batch[2 + threadSpotRow * 4][threadSpotCol * 3 + j] = __float2half(loadedModels[modelID].triangles[i * 20 + (threadSpotRow * 5 + threadSpotCol)].points[j].z);
                batch[3 + threadSpotRow * 4][threadSpotCol * 3 + j] = (half) 1;
            }
        }

        if (tid < 16) {
            batch[tid][15] = __float2half(0.0f); // pad the last column
        }
        
        __syncthreads();

        // wmma operations to multiply the matrices
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> d_frag;

        // Load matrix A and B from global memory
        wmma::load_matrix_sync(a_frag, &chunckedMatrix[0][0], 16);
        wmma::load_matrix_sync(b_frag, &batch[0][0], 16);

        // Load matrix C (accumulator)
        wmma::fill_fragment(c_frag, 0.0f);

        // Compute matrix multiply
        wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);

        Triangle* objTri = output[bid].triangles;

        if (triangleCount % 20 == 0 || i * 20 + threadSpotRow * 5 + threadSpotCol < triangleCount) {
            for (int j = 0; j < 4; j++) { // x, y, z, w components
                int triangleIndex = i * 20 + (threadSpotRow * 5 + threadSpotCol);
                int pointIndex = j; // Each row corresponds to one coordinate
                objTri[triangleIndex].points[pointIndex].x = __half2float(d_frag.x[(threadSpotRow * 4 + 0) * 16 + (threadSpotCol * 3 + j)]);
                objTri[triangleIndex].points[pointIndex].y = __half2float(d_frag.x[(threadSpotRow * 4 + 1) * 16 + (threadSpotCol * 3 + j)]);
                objTri[triangleIndex].points[pointIndex].z = __half2float(d_frag.x[(threadSpotRow * 4 + 2) * 16 + (threadSpotCol * 3 + j)]);
            }
        }
    }
}


__device__ void interpolatorUpdateHandler(interpolator* interp) {

}