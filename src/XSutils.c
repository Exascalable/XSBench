#include "XSbench_header.h"

// Allocates nuclide matrix
NuclideGridPoint ** gpmatrix(size_t m, size_t n)
{
	int i,j;
	NuclideGridPoint * full = (NuclideGridPoint *) malloc( m * n *
	                          sizeof( NuclideGridPoint ) );
	NuclideGridPoint ** M = (NuclideGridPoint **) malloc( m *
	                          sizeof(NuclideGridPoint *) );

	for( i = 0, j=0; i < m*n; i++ )
		if( i % n == 0 )
			M[j++] = &full[i];

	return M;
}

// Frees nuclide matrix
void gpmatrix_free( NuclideGridPoint ** M )
{
	free( *M );
	free( M );
}

// Compare function for two grid points. Used for sorting during init
int NGP_compare( const void * a, const void * b )
{
	NuclideGridPoint *i, *j;

	i = (NuclideGridPoint *) a;
	j = (NuclideGridPoint *) b;

	if( i->energy > j->energy )
		return 1;
	else if ( i->energy < j->energy)
		return -1;
	else
		return 0;
}

// Park & Miller Multiplicative Conguential Algorithm
// From "Numerical Recipes" Second Edition
double rn(unsigned long * seed)
{
	double ret;
	unsigned long n1;
	unsigned long a = 16807;
	unsigned long m = 2147483647;
	n1 = ( a * (*seed) ) % m;
	*seed = n1;
	ret = (double) n1 / m;
	return ret;
}


// RNG Used for Verification Option.
// This one has a static seed (must be set manually in source).
// Park & Miller Multiplicative Conguential Algorithm
// From "Numerical Recipes" Second Edition
double rn_v(void)
{
	static unsigned long seed = 1337;
	double ret;
	unsigned long n1;
	unsigned long a = 16807;
	unsigned long m = 2147483647;
	n1 = ( a * (seed) ) % m;
	seed = n1;
	ret = (double) n1 / m;
	return ret;
}

unsigned int hash(unsigned char *str, int nbins)
{
	unsigned int hash = 5381;
	int c;

	while (c = *str++)
		hash = ((hash << 5) + hash) + c;

	return hash % nbins;
}

size_t estimate_mem_usage( Inputs in )
{
	size_t single_nuclide_grid = in.n_gridpoints * sizeof( NuclideGridPoint );
	size_t all_nuclide_grids   = in.n_isotopes * single_nuclide_grid;
	size_t size_GridPoint      = sizeof(GridPoint) + in.n_isotopes*sizeof(int);
	size_t size_UEG            = in.n_isotopes*in.n_gridpoints * size_GridPoint;
	size_t memtotal;

	if( in.grid_type == UNIONIZED )
		memtotal          = all_nuclide_grids + size_UEG;
	else
		memtotal          = all_nuclide_grids;
	size_UEG          = size_UEG / 1048576;
	memtotal          = memtotal / 1048576;
	return memtotal;
}

void binary_dump(long n_isotopes, long n_gridpoints, NuclideGridPoint ** nuclide_grids, GridPoint * energy_grid, int grid_type)
{
	FILE * fp = fopen("XS_data.dat", "wb");
	// Dump Nuclide Grid Data
	for( long i = 0; i < n_isotopes; i++ )
		fwrite(nuclide_grids[i], sizeof(NuclideGridPoint), n_gridpoints, fp);

	if( grid_type == UNIONIZED )
	{
		// Dump UEG Data
		for( long i = 0; i < n_isotopes * n_gridpoints; i++ )
		{
			// Write energy level
			fwrite(&energy_grid[i].energy, sizeof(double), 1, fp);

			// Write index data array (xs_ptrs array)
			fwrite(energy_grid[i].xs_ptrs, sizeof(int), n_isotopes, fp);
		}
	}

	fclose(fp);
}

void binary_read(long n_isotopes, long n_gridpoints, NuclideGridPoint ** nuclide_grids, GridPoint * energy_grid, int grid_type)
{
	int stat;
	FILE * fp = fopen("XS_data.dat", "rb");
	// Read Nuclide Grid Data
	for( long i = 0; i < n_isotopes; i++ )
		stat = fread(nuclide_grids[i], sizeof(NuclideGridPoint), n_gridpoints, fp);

	if( grid_type == UNIONIZED )
	{
		// Dump UEG Data
		for( long i = 0; i < n_isotopes * n_gridpoints; i++ )
		{
			// Write energy level
			stat = fread(&energy_grid[i].energy, sizeof(double), 1, fp);

			// Write index data array (xs_ptrs array)
			stat = fread(energy_grid[i].xs_ptrs, sizeof(int), n_isotopes, fp);
		}
	}

	fclose(fp);

}

void application_checkpoint_read(int rank, int *bench, unsigned long long *vhash, Inputs *in, int *num_nucs, int **mats, double **concs) {
	char filename[20];
	FILE *fp;
	int last;
	sprintf(filename, "tmp_%d", rank);
	if ((fp = fopen(filename, "rb")) == NULL) {
		printf("ERROR: Opening file. Please check whether any checkpoint file exists.\n");
	}
	while(1) {
		fread(&last, sizeof(int), 1, fp);
		if(feof(fp)) break;
	}
	fclose(fp);

	sprintf(filename, "check_%d_%d", rank, last);
	if ((fp = fopen(filename, "rb")) == NULL) {
		printf("ERROR: Opening file to read. Please check whether any checkpoint file exists.\n");
	}
	fread(bench, sizeof(int), 1, fp);
	fread(vhash, sizeof(unsigned long long), 1, fp);

	fread(&(in->nthreads), sizeof(int), 1, fp);
	fread(&(in->n_isotopes), sizeof(long), 1, fp);
	fread(&(in->n_gridpoints), sizeof(long), 1, fp);
	fread(&(in->lookups), sizeof(int), 1, fp);
	fread(&(in->grid_type), sizeof(int), 1, fp);

	fread(num_nucs, sizeof(int), 12, fp);
	for (int i = 0; i < 12; i++) {
		mats[i] = (int *) malloc(num_nucs[i] * sizeof(int) );
		fread(mats[i], sizeof(int), num_nucs[i], fp);
	}

	for (int i = 0; i < 12; i++) {
		concs[i] = (double *)malloc( num_nucs[i] * sizeof(double) );
		fread(concs[i], sizeof(double), num_nucs[i], fp);
	}

	fclose(fp);
}

void application_checkpoint_write(int rank, int bench, unsigned long long vhash, Inputs in, int *num_nucs, int **mats, double **concs) {
	char filename[20];
	FILE *fp;
	sprintf(filename, "tmp_%d", rank);
	if ((fp = fopen(filename, "a")) == NULL) {
		printf("ERROR: Opening file\n");
	}
	fwrite(&bench, sizeof(int), 1, fp);
	fclose(fp);

	sprintf(filename, "check_%d_%d", rank, bench);

	if ((fp = fopen(filename, "wb")) == NULL) {
		printf("ERROR: Opening file to write\n");
	}

	fwrite(&bench, sizeof(int), 1, fp);
	fwrite(&vhash, sizeof(unsigned long long), 1, fp);

	fwrite(&in.nthreads, sizeof(int), 1, fp);
	fwrite(&in.n_isotopes, sizeof(long), 1, fp);
	fwrite(&in.n_gridpoints, sizeof(long), 1, fp);
	fwrite(&in.lookups, sizeof(int), 1, fp);
	fwrite(&in.grid_type, sizeof(int), 1, fp);

	fwrite(num_nucs, sizeof(int), 12, fp);
	for (int i = 0; i < 12; i++) {
		fwrite(mats[i], sizeof(int), num_nucs[i], fp);
	}

	for (int i = 0; i < 12; i++) {
		fwrite(concs[i], sizeof(double), num_nucs[i], fp);
	}
	fclose(fp);
}
