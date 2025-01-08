#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define XSIZE 2560
#define YSIZE 2048
#define MAXITER 255

double xleft = -2.01;
double xright = 1;
double yupper, ylower;
double ycenter = 1e-6;
double step;

int pixel[XSIZE * YSIZE];
#define PIXEL(i, j) ((i) + (j) * XSIZE)

typedef struct {
    double real, imag;
} complex_t;

void calculate() {
    for (int i = 0; i < XSIZE; i++) {
        for (int j = 0; j < YSIZE; j++) {
            complex_t c, z, temp;
            int iter = 0;
            c.real = (xleft + step * i);
            c.imag = (ylower + step * j);
            z = c;
            while (z.real * z.real + z.imag * z.imag < 4) {
                temp.real = z.real * z.real - z.imag * z.imag + c.real;
                temp.imag = 2 * z.real * z.imag + c.imag;
                z = temp;
                if (++iter == MAXITER) break;
            }
            pixel[PIXEL(i, j)] = iter;
        }
    }
}

void fancycolour(unsigned char *p, int iter) {
    if (iter == MAXITER);
    else if (iter < 8) {
        p[0] = 128 + iter * 16;
        p[1] = p[2] = 0;
    } else if (iter < 24) {
        p[0] = 255;
        p[1] = p[2] = (iter - 8) * 16;
    } else if (iter < 160) {
        p[0] = p[1] = 255 - (iter - 24) * 2;
        p[2] = 255;
    } else {
        p[0] = p[1] = (iter - 160) * 2;
        p[2] = 255 - (iter - 160) * 2;
    }
}

void savebmp(char *name, unsigned char *buffer, int x, int y) {
    FILE *f = fopen(name, "wb");
    if (!f) {
        printf("Error writing image to disk.\n");
        return;
    }
    unsigned int size = x * y * 3 + 54;
    unsigned char header[54] = {'B', 'M', size & 255, (size >> 8) & 255, (size >> 16) & 255, size >> 24, 0,
                                0, 0, 0, 54, 0, 0, 0, 40, 0, 0, 0, x & 255, x >> 8, 0, 0, y & 255, y >> 8, 0, 0,
                                1, 0, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    fwrite(header, 1, 54, f);
    fwrite(buffer, 1, XSIZE * YSIZE * 3, f);
    fclose(f);
}

int main(int argc, char **argv) {
    clock_t start_time, end_time;
    double total_time;

    // Start timing
    start_time = clock();

    step = (xright - xleft) / XSIZE;
    yupper = ycenter + (step * YSIZE) / 2;
    ylower = ycenter - (step * YSIZE) / 2;

    calculate();

    if (strtol(argv[1], NULL, 10) != 0) {
        // Create and save the image
        unsigned char *buffer = calloc(XSIZE * YSIZE * 3, 1);
        for (int i = 0; i < XSIZE; i++) {
            for (int j = 0; j < YSIZE; j++) {
                int p = ((YSIZE - j - 1) * XSIZE + i) * 3;
                fancycolour(buffer + p, pixel[PIXEL(i, j)]);
            }
        }
        savebmp("mandel2.bmp", buffer, XSIZE, YSIZE);
    }

    // End timing
    end_time = clock();
    total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    printf("Total time spent: %f seconds.\n", total_time);

    return 0;
}
