#include <GL/glut.h> 
#include <stdlib.h> 
#include <stdio.h>
#include <windows.h>

#pragma warning (disable: 4996)

#define MAX_TIMEFRAME 2000

float* pMat, * pMatRes;
int nrows, ncols, timeframe = 0, bOk = 0, complete = 0;
FILE* fp;

void idle() {
    glutPostRedisplay();
}

void magic_dots(void) {
    float r, g, b, ratio;
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0, nrows, 0.0, ncols);
    if (timeframe < MAX_TIMEFRAME & bOk == 0) {
        timeframe++;
        int i = 2, j = 0;
        while (i < nrows) {
            int posRes = (i * ncols) - 2;
            int posA = ((i - 1) * ncols) - 3;
            int posB = ((i - 1) * ncols) - 2;
            int posC = ((i - 1) * ncols) - 1;
            int posD = (i * ncols) - 3;
            int posE = (i * ncols) - 1;
            int posF = ((i + 1) * ncols) - 3;
            int posG = ((i + 1) * ncols) - 2;
            int posH = ((i + 1) * ncols) - 1;
            for (j = posRes; j > (posRes - ncols) + 2; j--) {
                pMatRes[j] = (pMat[posA--] + pMat[posB--] + pMat[posC--] + pMat[posD--] + pMat[posE--] + pMat[posF--] + pMat[posG--] + pMat[posH--] + pMat[j]) / 9;
            }
            i++;
        }
        memcpy(pMat, pMatRes, nrows * ncols * sizeof(float));

        //Assign RGB in each spot
        for (int i = 0; i < nrows; i++) {
            for (int j = 0; j < ncols; j++) {
                ratio = 2 * (pMat[(i * ncols) + j]) / 255;
                b = 1 - ratio;
                if (b < 0)
                    b = 0;
                r = ratio - 1;
                if (r < 0)
                    r = 0;
                g = 1 - b - r;
                glColor3f(r, g, b);
                glBegin(GL_POINTS);
                glVertex2i(i, j);
                glEnd();
            }
        }
    }
    else if (timeframe == MAX_TIMEFRAME & complete == 0 & bOk == 0)
        bOk = 1;
    else if(bOk == 1 & complete == 0) {
        glutSetWindowTitle("[Completed!] Heat Transfer with OpenGL (60070503466 & 60070503482)");
        complete = 1;
    }
    glFlush();
}


int main(int argc, char* argv[]) {

    fp = fopen("heatMatrixGL.txt", "r");
    int i = 0;
    fscanf(fp, "%d %d", &nrows, &ncols);
    pMat = (float*)calloc(nrows * ncols, sizeof(float));
    pMatRes = (float*)calloc(nrows * ncols, sizeof(float));
    while (fscanf(fp, "%f", &pMat[i++]) == 1);
    fclose(fp);
    memcpy(pMatRes, pMat, nrows * ncols * sizeof(float));

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE);
    glutInitWindowSize(nrows, ncols);
    glutCreateWindow("[Running] Heat Transfer with OpenGL (60070503466 & 60070503482)");
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glutDisplayFunc(magic_dots);
    glutIdleFunc(idle);
    glutMainLoop();

    return 0;
}