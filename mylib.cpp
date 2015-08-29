#include "mylib.h"

float mysum(int rows, int cols, float *array) {
    float sum = 0;
    for(int row = 0; row < rows; row++) {
        for(int col = 0; col < cols; col++) {
            sum += array[row * cols + col];
        }
    }
    return sum;
}

