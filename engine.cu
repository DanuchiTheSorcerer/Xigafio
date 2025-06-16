#include "engine.h"
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <windows.h>
#include <vector>

using namespace nvcuda;

constexpr float pi = 3.14159265358979323846f; // Define pi as a constant
constexpr int maxAmountOfModels = 256*1024; // maximum amount of models device will store
constexpr int maxAmountOfTriangles = 64*1024*1024; // maximum amount of triangles that can be rendered
constexpr int maxAmountOfMeshes = 1024*1024; // maximum amount of triangles that can be rendered
Model* models; // device pointer to array of models
Triangle** triAllocs; // host pointer to array of dev ptrs to triangle arrays, obtained from allocation, for freeing
std::vector<int> usedModelIDs; // list of model ids allocated to, used for freeing
bool loadingModels; // meant to keep things synchronized when loading models
Triangle* scene; // array of triangles before rasterization
Mesh* meshes; // dev pointer to models for the tick
int meshCountThisTick;
Mesh* lastTickMeshes; // dev pointer to last tick's models; used for interpolation
Mesh* meshBuffer; // cpu copys to it (works like extension of interp buffer behavior)

// Frees all models data and sub allocations, but not models itself
void clearModels() {
    for (int i = 0; i < usedModelIDs.size(); i++) {
        cudaFree(triAllocs[i]);
    }
    usedModelIDs.clear();
}

// Allocates space for triangles on device and copys data over to device
// @param ID id of the model being loaded
// @param triangles host pointer to array of triangles
// @param triangleCount number of triangles in array
// @note Only works if loadingModels is true to avoid desync. \n
//  It is also highly recommended that the Mesh allocation be made BEFORE loading models to avoid cheese memory
void loadModel(int ID, Triangle* triangles, int triangleCount) {
    if (loadingModels) {
        Model model;
        model.triangleCount = triangleCount;
        cudaMalloc(&model.triangles,sizeof(Triangle)*triangleCount);
        // copy model to device
        cudaMemcpy(&models[ID],&model,sizeof(Model),cudaMemcpyHostToDevice);
        // copy triangles to device
        cudaMemcpy(model.triangles,triangles,sizeof(Triangle)*triangleCount,cudaMemcpyHostToDevice);
        usedModelIDs.push_back(ID);
    }
};

// Copies a mesh to the mesh buffer on device
// @param allOfTheAbove just the elements of the mesh being copied
void loadMesh(int modelID,float3 pos,float3 rotAxis, float scale, float theta) {
    Mesh mesh;
    mesh.modelID = modelID;
    mesh.pos = pos;
    mesh.rotAxis = rotAxis;
    mesh.scale = scale;
    mesh.theta = theta;

    cudaMemcpy(meshBuffer + meshCountThisTick,&mesh,sizeof(Mesh),cudaMemcpyHostToDevice);
    meshCountThisTick++;
}

interpolator tickLogic(int tickCount) {
    interpolator result;
    meshCountThisTick = 0;

    if (tickCount == 0) {
        cudaMalloc(&scene,sizeof(Triangle) * maxAmountOfTriangles);
        cudaMalloc(&meshes,sizeof(Mesh) * maxAmountOfMeshes);
        cudaMalloc(&lastTickMeshes,sizeof(Mesh) * maxAmountOfMeshes);
        cudaMalloc(&models, sizeof(Model) * maxAmountOfModels);
        triAllocs = (Triangle**) malloc(sizeof(Triangle*) * maxAmountOfModels);
    }

    Model tm;
    tm.triangles = new Triangle[1];
    tm.triangles[0].p1.x = 0.0f;
    tm.triangles[0].p1.y = 0.0f;
    tm.triangles[0].p1.z = 0.0f;
    tm.triangles[0].p2.x = 1.0f;
    tm.triangles[0].p2.y = 0.0f;
    tm.triangles[0].p2.z = 0.0f;
    tm.triangles[0].p3.x = 0.0f;
    tm.triangles[0].p3.y = 1.0f;
    tm.triangles[0].p3.z = 0.0f;


    Camera camera;
    camera.pos = make_float3(0.0f, 0.0f, 1.0f);
    camera.rotAxis = make_float3(1.0f,0.0f,0.0f);
    camera.theta = pi;
    camera.focalLength = 1.0f;


    if (tickCount == 0) {
        loadingModels = true;
        loadModel(0,tm.triangles,1);
    }


    if (tickCount == 200) {
        loadingModels = false;
    }

    result.tickCount = tickCount;
    result.models = models;
    result.loadingModels = loadingModels;
    result.camera = camera;
    result.scene = scene;
    result.meshBuffer = meshBuffer;
    result.meshes = meshes;
    result.lastTickMeshes = lastTickMeshes;
    result.bufferMeshCount = meshCountThisTick;
    return result;
};








__global__ void computePixel(uint32_t* buffer, int width, int height, const interpolator* interp,float inpf) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (interp->loadingModels) {
        if (x < width && y < height) {
            int idx = y * width + x;
            int speed = 3;
            int modVal = interp->tickCount*speed;
            uint32_t red   = 128 + 127*sinf((float)x/128 + ((float)modVal+inpf*speed)/60);
            uint32_t green = 128 + 127*cosf((float)y/128 + ((float)modVal+inpf*speed)/60);
            uint32_t blue  = 128 + 12*sinf((float)x/128 + (float)y/128 + ((float)modVal+inpf*speed)/6);
            buffer[idx] = 0xFF000000 | (red << 16) | (green << 8) | blue;
        }
    } else {
        if (x < width && y < height) {
            int idx = y * width + x;
            uint32_t blah;
            int modelID = 0;
            int tn = 0;
            if (interp->models[modelID].triangleCount != 0 && x == 100) {
                blah = 0;
            } else if (interp->models[modelID].triangles[tn].p1.x != 0 && x == 150) {
                blah = 0;
            } else if (interp->models[modelID].triangles[tn].p1.y != 0 && x == 200) {
                blah = 0;
            } else if (interp->models[modelID].triangles[tn].p1.z != 0 && x == 250) {
                blah = 0;
            } else if (interp->models[modelID].triangles[tn].p2.x != 0 && x == 300) {
                blah = 0;
            } else if (interp->models[modelID].triangles[tn].p2.y != 0 && x == 350) {
                blah = 0;
            } else if (interp->models[modelID].triangles[tn].p2.z != 0 && x == 400) {
                blah = 0;
            } else if (interp->models[modelID].triangles[tn].p3.x != 0 && x == 450) {
                blah = 0;
            } else if (interp->models[modelID].triangles[tn].p3.y != 0 && x == 500) {
                blah = 0;
            } else if (interp->models[modelID].triangles[tn].p3.z != 0 && x == 550) {
                blah = 0;
            } else {
                blah = 255;
            }
            buffer[idx] = 0xFF000000 | (blah << 16) | (blah << 8) | blah;
        }
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



__global__ void meshesToWorld(interpolator* interp) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    

    __shared__ half M16[16][16];

    // Only one thread per block needs to compute M16
    if (tid == 0) {

        float3 U1 = interp->meshes[bid].rotAxis;
        float THETA1 = interp->meshes[bid].theta;
        float S = interp->meshes[bid].scale;
        float3 T1 = interp->meshes[bid].pos;
        float3 T2 = interp->camera.pos;
        float3 U2 = interp->camera.rotAxis;
        float THETA2 = interp->camera.theta;
        float focalLength = interp->camera.focalLength;

        float M4[4][4];

        // 1) Normalize the rotation axes
        float invLen1 = rsqrtf(U1.x*U1.x + U1.y*U1.y + U1.z*U1.z);
        U1.x *= invLen1;  U1.y *= invLen1;  U1.z *= invLen1;
        float invLen2 = rsqrtf(U2.x*U2.x + U2.y*U2.y + U2.z*U2.z);
        U2.x *= invLen2;  U2.y *= invLen2;  U2.z *= invLen2;

        // 2) Build R1 = R(U1, +THETA1)
        float c1 = cosf(THETA1), s1 = sinf(THETA1), C1 = 1.0f - c1;
        float R1[3][3] = {
            { c1 + U1.x*U1.x*C1,    U1.x*U1.y*C1 - U1.z*s1,  U1.x*U1.z*C1 + U1.y*s1 },
            { U1.y*U1.x*C1 + U1.z*s1, c1 + U1.y*U1.y*C1,     U1.y*U1.z*C1 - U1.x*s1 },
            { U1.z*U1.x*C1 - U1.y*s1, U1.z*U1.y*C1 + U1.x*s1, c1 + U1.z*U1.z*C1     }
        };

        // 3) Build R2 = R(U2, -THETA2) for left‑handed rotation
        float c2 = cosf(THETA2), s2 = -sinf(THETA2), C2 = 1.0f - c2;
        float R2[3][3] = {
            { c2 + U2.x*U2.x*C2,    U2.x*U2.y*C2 - U2.z*s2,  U2.x*U2.z*C2 + U2.y*s2 },
            { U2.y*U2.x*C2 + U2.z*s2, c2 + U2.y*U2.y*C2,     U2.y*U2.z*C2 - U2.x*s2 },
            { U2.z*U2.x*C2 - U2.y*s2, U2.z*U2.y*C2 + U2.x*s2, c2 + U2.z*U2.z*C2     }
        };

        // 4) Combine R = R2 * R1
        float R[3][3];
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                R[i][j] = R2[i][0]*R1[0][j]
                        + R2[i][1]*R1[1][j]
                        + R2[i][2]*R1[2][j];
            }
        }

        // 5) Compute the net translation t = R2*T1 - R2*T2
        float3 RT1 = make_float3(
            R2[0][0]*T1.x + R2[0][1]*T1.y + R2[0][2]*T1.z,
            R2[1][0]*T1.x + R2[1][1]*T1.y + R2[1][2]*T1.z,
            R2[2][0]*T1.x + R2[2][1]*T1.y + R2[2][2]*T1.z
        );
        float3 RT2 = make_float3(
            R2[0][0]*T2.x + R2[0][1]*T2.y + R2[0][2]*T2.z,
            R2[1][0]*T2.x + R2[1][1]*T2.y + R2[1][2]*T2.z,
            R2[2][0]*T2.x + R2[2][1]*T2.y + R2[2][2]*T2.z
        );
        float3 t = make_float3(RT1.x - RT2.x,
                               RT1.y - RT2.y,
                               RT1.z - RT2.z);

        // 6) Fill M4 = M3 * (M2*M1) with a 4th zero‐row
        //    M4 is 4×4; row 0..2 come from the 3×4 M3*(M2*M1), row 3 is all zeros
        M4[0][0] = focalLength * S * R[0][0];
        M4[0][1] = focalLength * S * R[0][1];
        M4[0][2] = focalLength * S * R[0][2];
        M4[0][3] = focalLength *       t.x;

        M4[1][0] = focalLength * S * R[1][0];
        M4[1][1] = focalLength * S * R[1][1];
        M4[1][2] = focalLength * S * R[1][2];
        M4[1][3] = focalLength *       t.y;

        M4[2][0] =            S * R[2][0];
        M4[2][1] =            S * R[2][1];
        M4[2][2] =            S * R[2][2];
        M4[2][3] =                  t.z;

        // 4th row = all zeros
        for (int j = 0; j < 4; ++j) {
            M4[3][j] = 0.0f;
        }

        // Fill M16 with 4 M4s on its diagonal, 0s elsewhere
        // M16 is 16x16, M4 is 4x4
        for (int block = 0; block < 4; ++block) {
            for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                int row = block * 4 + i;
                int col = block * 4 + j;
                M16[row][col] = __float2half(M4[i][j]);
            }
            }
        }
        // Set all other elements to 0
        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < 16; ++j) {
            // If not on one of the 4 M4 diagonal blocks, set to 0
            if (!((i / 4 == j / 4) && (i % 4 < 4) && (j % 4 < 4))) {
                M16[i][j] = __float2half(0.0f);
            }
            }
        }
    }

    // Make sure all threads see the finished M16
    __syncthreads();

    // ... now every thread in the block can use M16 ...

    //for all threads to use
    __shared__ half V16[16][16];
    __shared__ Model* model;
    __shared__ int totalTriangles;

    if (tid == 0) {
        model = &(interp->models[interp->meshes[bid].modelID]);
        totalTriangles = model->triangleCount;
    }
    __syncthreads();
        
    // can only batch 20 triangles at a time, so loop until all vectors have been processed
    for (int batch = 0; batch < ((totalTriangles + 19) / 20); batch++) {
        //  ... STEPS ...
        // threads work together to fill V16
        // syncthreads
        // WMMA MatMul to produce transformed vectors
        // extract vectors 
        int trianglesThisBatch = min(20, totalTriangles - batch * 20);


    }
}



__device__ void interpolatorUpdateHandler(interpolator* interp) {
    if (interp->tickCount == 0) {
    }

    //copy the mesh to last mesh
    for (int i = 0; i < interp->meshesCount;i++) {
        *(lastTickMeshes+i) = *(meshes+i);
    }
    interp->lastTickMeshCount = interp->meshesCount;

    //copy buffer mesh to mesh
    for (int i = 0;i < interp->bufferMeshCount;i++) {
        *(meshes+i) = *(meshBuffer + i);
    }
    interp->meshesCount = interp->bufferMeshCount;
}


void cleanUpCall() {
    clearModels();
    cudaFree(models);
    cudaFree(scene);
    cudaFree(meshBuffer);
    cudaFree(meshes);
    cudaFree(lastTickMeshes);
    free(triAllocs);
    usedModelIDs.clear();
}