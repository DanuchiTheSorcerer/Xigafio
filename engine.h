#ifndef ENGINE_H
#define ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

// A simple structure holding game state for interpolation.
// Extend this structure with more state as needed.

struct Point {
    float x;
    float y;
    float z;
};

struct Triangle {
    Point points[3];
};

struct Mesh {
    Triangle* triangles;
    int triangleCount;
};

struct Model {
    Triangle* triangles;
    int triangleCount;
    float scale;
    float rotation[4] = {0,0,0,0};
    float position[3] = {0,0,0};
};

struct interpolator  {
    int tickCount;
    Mesh mesh;
    Model* models;
};


// tickLogic
// Processes game logic for the given tick count and returns an interpolator 
// containing the updated game state.
interpolator tickLogic(int tickCount);

//compute a frame
__device__ void computeFrame(uint32_t* buffer, int width, int height, const interpolator* interp,float interpolationFactor);

//put gpu response to new interpolator
__device__ void interpolatorUpdateHandler();

#ifdef __cplusplus
}
#endif

#endif ENGINE_H
