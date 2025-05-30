#ifndef ENGINE_H
#define ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

// A simple structure holding game state for interpolation.
// Extend this structure with more state as needed.

// struct Camera {
//     float x;
//     float y;
//     float z;
//     float rotX;
//     float rotY;
//     float rotZ;
//     float theta;
// };

struct Triangle {
    float x1;
    float y1;
    float z1;
    float x2;
    float y2;
    float z2;
    float x3;
    float y3;
    float z3;
};

struct Model {
    Triangle* triangles;
    int triangleCount;
};

// struct Mesh {
//     int modelID;
//     float scale;
//     float x;
//     float y;
//     float z;
//     float rotX;
//     float rotY;
//     float rotZ;
//     float theta;
//     bool dynamic; // set to true only if the mesh ever moves and/or rotates
// };

struct interpolator  {
    int tickCount;
    Model* models;
    bool loadingModels;
};


// tickLogic
// Processes game logic for the given tick count and returns an interpolator 
// containing the updated game state.
interpolator tickLogic(int tickCount);

//compute a frame
__device__ void computeFrame(uint32_t* buffer, int width, int height, const interpolator* interp,float interpolationFactor);

//put gpu response to new interpolator
__device__ void interpolatorUpdateHandler(interpolator* interp);

//clean up for use by game engine user
void cleanUpCall();

#ifdef __cplusplus
}
#endif

#endif ENGINE_H
