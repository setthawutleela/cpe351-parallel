/* lab1.c
    Created by  Setthawut Leelawatthanapanit    ID: 60070503466
               Naphatthorn Kayanthanakorn      ID: 60070503482
    Date: 24/01/2020
*/

#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]){
    int i;
    int id;
    int p;
    int j = 1;    // For loop

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    
    if(id != 0)
        MPI_Send(&id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    else{
        while(j < p){
            MPI_Recv(&i, 1, MPI_INT, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Hello Rank %d, I'm rank %d.\n", id, i);
            fflush(stdin);
            j++;
        }
    }
    MPI_Finalize();
}