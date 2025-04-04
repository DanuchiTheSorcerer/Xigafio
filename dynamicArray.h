#ifndef CUDA_DYNAMIC_ARRAY_CUH
#define CUDA_DYNAMIC_ARRAY_CUH

#include <cuda_runtime.h>
#include <stdio.h>

namespace cuda_containers {

    /**
     * @brief Device-side dynamic array implementation with automatic resizing
     * 
     * This class provides a vector-like container that operates entirely on the 
     * GPU device without requiring host intervention for memory management.
     * 
     * @tparam T Type of elements stored in the array
     */
    template <typename T>
    class DynamicArray {
    public:
        /**
         * @brief Default constructor
         * 
         * Initializes an empty array with zero capacity.
         * Call initialize() before using.
         */
        __device__ DynamicArray() : data(NULL), size(0), capacity(0), growthFactor(1.5f) {}
        
        /**
         * @brief Destructor - frees device memory
         */
        __device__ ~DynamicArray() {
            if (data != NULL) {
                free(data);
                data = NULL;
            }
        }
        
        /**
         * @brief Initialize array with given capacity
         * 
         * @param initialCapacity Initial capacity of the array
         * @param growth Factor by which capacity increases (default: 1.5)
         * @return true if initialization successful, false otherwise
         */
        __device__ bool initialize(size_t initialCapacity, float growth = 1.5f) {
            if (data != NULL) {
                free(data); // Free existing data if already initialized
            }
            
            growthFactor = growth > 1.0f ? growth : 1.5f; // Ensure valid growth factor
            
            if (initialCapacity > 0) {
                data = (T*)malloc(initialCapacity * sizeof(T));
                if (data == NULL) return false;
                capacity = initialCapacity;
            } else {
                data = NULL;
                capacity = 0;
            }
            
            size = 0;
            return true;
        }
        
        /**
         * @brief Adds an element to the end of the array
         * 
         * Automatically resizes if necessary according to growth factor.
         * 
         * @param element Element to add
         * @return true if successful, false if memory allocation failed
         */
        __device__ bool push_back(const T& element) {
            if (size >= capacity) {
                // Calculate new capacity with growth factor
                size_t newCapacity = max(1, (size_t)(capacity * growthFactor));
                if (newCapacity <= capacity) {
                    newCapacity = capacity + 1; // Ensure we always grow
                }
                
                // Resize array
                T* newData = (T*)resizeArray(data, size * sizeof(T), newCapacity * sizeof(T));
                if (newData == NULL) return false;
                
                data = newData;
                capacity = newCapacity;
            }
            
            // Add the new element
            data[size++] = element;
            return true;
        }
        
        /**
         * @brief Removes the last element from the array
         * 
         * @return true if successful, false if array was empty
         */
        __device__ bool pop_back() {
            if (size > 0) {
                size--;
                return true;
            }
            return false;
        }
        
        /**
         * @brief Clears all elements without changing capacity
         */
        __device__ void clear() {
            size = 0;
        }
        
        /**
         * @brief Explicitly resizes the array
         * 
         * @param newCapacity New capacity for the array
         * @return true if successful, false if memory allocation failed
         */
        __device__ bool resize(size_t newCapacity) {
            if (newCapacity == capacity) return true;
            
            // Special case: resize to zero
            if (newCapacity == 0) {
                if (data != NULL) {
                    free(data);
                    data = NULL;
                }
                size = 0;
                capacity = 0;
                return true;
            }
            
            // Resize array with new capacity
            T* newData = (T*)resizeArray(data, 
                                        min(size, newCapacity) * sizeof(T), 
                                        newCapacity * sizeof(T));
            if (newData == NULL) return false;
            
            data = newData;
            capacity = newCapacity;
            
            // Adjust size if new capacity is smaller
            if (size > capacity) {
                size = capacity;
            }
            
            return true;
        }
        
        /**
         * @brief Access element by index with bounds checking
         * 
         * @param index Index of element to access
         * @param success Optional pointer to store success status
         * @return Reference to the element or first element if out of bounds
         */
        __device__ T& at(size_t index, bool* success = nullptr) {
            if (index < size) {
                if (success) *success = true;
                return data[index];
            }
            
            if (success) *success = false;
            return data[0]; // Return first element as fallback
        }
        
        /**
         * @brief Access element by index without bounds checking
         * 
         * @param index Index of element to access
         * @return Reference to the element
         */
        __device__ T& operator[](size_t index) {
            return data[index];
        }
        
        /**
         * @brief Get the current number of elements
         * 
         * @return Size of array
         */
        __device__ size_t get_size() const {
            return size;
        }
        
        /**
         * @brief Get the current capacity of the array
         * 
         * @return Capacity of array
         */
        __device__ size_t get_capacity() const {
            return capacity;
        }
        
        /**
         * @brief Get pointer to raw data array
         * 
         * @return Pointer to internal data
         */
        __device__ T* data_ptr() {
            return data;
        }
    
    private:
        T* data;              // Pointer to array data
        size_t size;          // Current number of elements
        size_t capacity;      // Total capacity
        float growthFactor;   // Factor by which to grow when resizing
    
        // Helper function to resize array
        __device__ void* resizeArray(void* oldArray, size_t oldSize, size_t newSize) {
            // Allocate new memory on device
            void* newArray = malloc(newSize);
            
            if (newArray == NULL) {
                return NULL; // Allocation failed
            }
            
            // Copy data from old array to new array if it exists
            if (oldArray != NULL && oldSize > 0) {
                memcpy(newArray, oldArray, oldSize);
                free(oldArray); // Free old memory
            }
            
            return newArray;
        }
    };
    
    } // namespace cuda_containers

#endif // CUDA_DYNAMIC_ARRAY_CUH