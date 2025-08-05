#include <iostream>
#include <cstdlib> // For atoi, rand, srand
#include <ctime>   // For time
#include <mpi.h>

void walker_process();
void controller_process();

int domain_size;
int max_steps;
int world_rank;
int world_size;

int main(int argc, char **argv)
{
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes and the rank of this process
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc != 3)
    {
        if (world_rank == 0)
        {
            std::cerr << "Usage: mpirun -np <p> " << argv[0] << " <domain_size> <max_steps>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    domain_size = atoi(argv[1]);
    max_steps = atoi(argv[2]);

    if (world_rank == 0)
    {
        // Rank 0 is the controller
        controller_process();
    }
    else
    {
        // All other ranks are walkers
        walker_process();
    }

    // Finalize the MPI environment
    MPI_Finalize();
    return 0;
}

void walker_process()
{
    // Seed the random number generator.
    // Using rank ensures each walker gets a different sequence of random numbers.
    srand(time(NULL) + world_rank);

    int position = 0; // Walk starts from origin
    int steps = 0;    // Step counter

    // Perform random walk until walker exits domain or hits max_steps
    while (std::abs(position) <= domain_size && steps < max_steps)
    {
        // Randomly choose direction: -1 (left) or +1 (right)
        int direction = (rand() % 2 == 0) ? -1 : 1;
        position += direction;
        steps++;
    }

    // Report how many steps this walker took before stopping
    std::cout << "Rank " << world_rank << ": Walker finished in " << steps << " steps." << std::endl;

    // Notify controller (rank 0) that this walker has finished
    int done_signal = 1;
    MPI_Send(&done_signal, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
}

void controller_process()
{
    int completed_walkers = 0;
    int total_walkers = world_size - 1;

    // Wait until all walkers have sent their completion signal
    while (completed_walkers < total_walkers)
    {
        MPI_Status status;

        // Wait for any incoming message from any walker
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        int count;
        // Get number of elements in the incoming message
        MPI_Get_count(&status, MPI_INT, &count);

        if (count == 1)
        {
            int message;
            // Receive the message and update completed count
            MPI_Recv(&message, count, MPI_INT,
                     status.MPI_SOURCE, status.MPI_TAG,
                     MPI_COMM_WORLD, &status);
            completed_walkers++;
        }
    }

    // All walkers have completed; print summary
    std::cout << "Controller: All " << total_walkers << " walkers have finished." << std::endl;
}
