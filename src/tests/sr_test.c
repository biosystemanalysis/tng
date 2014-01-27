#ifdef TNG_BUILD_OPENMP_EXAMPLES

#include "tng/tng_io.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <unistd.h>
#include <sys/utsname.h> // for uname
#include <time.h>
int main();
void compute(int number_of_particles, int dimensions, double positions[], double vel[], double mass, double forces[], double *pot, double *kin);
double dist(int dimensions, double r1[], double r2[], double dr[]);
void initialize(int particle_count, int num_dims, double box[], double pos[], double vel[], double acc[]);
void timestamp(void);
void update(int number_of_particles, int dimensions, double *box, double pos[], double vel[], double foces[], double acc[], double mass, double dt);

void fail(tng_trajectory_t *traj, int code) {
	fprintf(stderr,"...failed! => %d\n",code);
	tng_trajectory_destroy(traj);
	exit(code);
}

void fail2(tng_trajectory_t *traj1,tng_trajectory_t *traj2, int code) {
	fprintf(stderr,"...failed! => %d\n",code);
	tng_trajectory_destroy(traj1);
	tng_trajectory_destroy(traj2);
	exit(code);
}

void greeter(int number_of_particles, int step_num, double dt, int proc_num, int threads, int dimensions, double *box) {
	printf("\n");
	printf("MD_OPENMP\n");
	printf("  C/OpenMP version\n");
	printf("\n");
	printf("  A molecular dynamics program.\n");

	printf("\n");
	printf("  NP, the number of particles in the simulation is %d\n", number_of_particles);
	printf("  STEP_NUM, the number of time steps, is %d\n", step_num);
	printf("  DT, the size of each time step, is %f\n", dt);


	printf("\n");
	printf("  Box shape: %f", box[0]);
	int dim = 1;
	for (; dim < dimensions; dim++) {
		printf(" x %f", box[dim]);
	}

	printf("\n");
}

/* this creates 2/3 * number_of_particles molecules of water and 1/3 molecles of love. */
void create_molecules(tng_trajectory_t *traj, const int number_of_particles) {
	tng_molecule_t water;
	tng_chain_t water_chain;
	tng_residue_t water_chain_residue;
	tng_atom_t water_residue_atom;

	/* Set molecules data */
	printf("  Creating molecules in trajectory.\n");

	if (tng_molecule_add(*traj, "water", &water) != TNG_SUCCESS) fail(traj,__LINE__);
	if (tng_molecule_chain_add(*traj, water, "W", &water_chain) != TNG_SUCCESS) fail(traj,__LINE__); // molecule, name
	if (tng_chain_residue_add(*traj, water_chain, "WAT", &water_chain_residue) != TNG_SUCCESS) fail(traj,__LINE__);; // chain, name
	if (tng_residue_atom_add(*traj, water_chain_residue, "Hydrogen", "Hydrogen Type", &water_residue_atom) != TNG_SUCCESS) fail(traj,__LINE__); //atom name,atom type, *atom
	if (tng_molecule_cnt_set(*traj, water, 2 * number_of_particles / 3) != TNG_SUCCESS) fail(traj,__LINE__);

	tng_molecule_t love;
	tng_chain_t love_chain;
	tng_residue_t love_chain_residue;
	tng_atom_t love_residue_atom;

	if (tng_molecule_add(*traj, "love", &love) != TNG_SUCCESS) fail(traj,__LINE__); // name
	if (tng_molecule_chain_add(*traj, love, "W", &love_chain) != TNG_SUCCESS) fail(traj,__LINE__); // molecule, name
	if (tng_chain_residue_add(*traj, love_chain, "WAT", &love_chain_residue) != TNG_SUCCESS) fail(traj,__LINE__); // chain, name
	if (tng_residue_atom_add(*traj, love_chain_residue, "love atom", "love atom type", &love_residue_atom) != TNG_SUCCESS) fail(traj,__LINE__);
	if (tng_molecule_cnt_set(*traj, love, number_of_particles - (2 * number_of_particles / 3)) != TNG_SUCCESS) fail(traj,__LINE__);
}

void lammpstrj_write_positions(FILE *f, const int64_t frame_nr, int dimensions, double *box, const int64_t number_of_particles, double *values) {
	fprintf(f, "ITEM: TIMESTEP\n%d\n", frame_nr);
	fprintf(f, "ITEM: NUMBER OF ATOMS\n%d\n", number_of_particles);
	fprintf(f, "ITEM: BOX BOUNDS\n");
	int dimension = 0;
	for (; dimension < dimensions; dimension++) {
		fprintf(f, "0 %f\n", box[dimension]);
	}
	fprintf(f, "ITEM: ATOMS id type x y z\n");
	int atom = 0;
	for (; atom < number_of_particles; atom++) {
		fprintf(f, "%d ", 1 + atom);
		fprintf(f, "%d", atom % 7);
		for (dimension = 0; dimension < dimensions; dimension++) {
			int index = dimensions * atom + dimension;
			fprintf(f, " %f", values[dimensions * atom + dimension]);
		}
		fprintf(f, "\n");
	}
}


void free_structures(double* d1, double* d2, double* d3, double* d4, double* d5) {
	free(d1);
	free(d2);
	free(d3);
	free(d4);
	free(d5);
}

void print_time(){
	time_t rawtime;
	time ( &rawtime );
	printf ("%d",rawtime);
}

double random_value(double max) {
	return max * rand() / (RAND_MAX + 1.0);
}

char *create_trajectory(int particle_count, int step_count, int initial_offset, int step_save, double dt, int proc_count, int thread_count, int dimensions, double *box,int dst_frames_per_block,char *filename,int save,int64_t codec_id, double precision) {
	/* Start initialization */

	char hash_mode = TNG_USE_HASH;

	printf("Initializing trajectory storage:\n");
	tng_trajectory_t sink_traj; // sink trajectory handle
	if (tng_trajectory_init(&sink_traj) != TNG_SUCCESS) fail(&sink_traj, __LINE__);

	int64_t medium_stride_length = 5;

	if (tng_medium_stride_length_set(sink_traj, medium_stride_length) != TNG_SUCCESS) fail(&sink_traj,__LINE__);
	printf("\t- medium stride length: %d\n", medium_stride_length);

	int64_t long_stride_length = 25;

	if (tng_long_stride_length_set(sink_traj, 25) != TNG_SUCCESS) fail(&sink_traj,__LINE__);
	printf("\t- long stride length: %d\n", long_stride_length);

	if (tng_num_frames_per_frame_set_set(sink_traj,dst_frames_per_block) != TNG_SUCCESS) fail(&sink_traj,__LINE__);
	printf("\t- number of frames per block (sink): %d.\n",dst_frames_per_block);

	if (tng_compression_precision_set(sink_traj, 1/precision) != TNG_SUCCESS) fail(&sink_traj, __LINE__);
	printf("\t- set compression precision: %f\n", precision);

	char *username = getenv("USER");
	if (username == NULL) fail(&sink_traj, 3);
	printf("\t- writing user data");
	if (tng_first_user_name_set(sink_traj, username) != TNG_SUCCESS) fail(&sink_traj,__LINE__);

	char *prog_name = getenv("_");
	if (prog_name == NULL) fail(&sink_traj, 4);
	if (tng_first_program_name_set(sink_traj, prog_name) != TNG_SUCCESS) fail(&sink_traj,__LINE__);

	struct utsname buffer;
	if (uname(&buffer) != 0) fail(&sink_traj, 2);
	printf(".\n\t- writing computer data");

	if (tng_first_computer_name_set(sink_traj, buffer.nodename) != TNG_SUCCESS) fail(&sink_traj,__LINE__);

	printf("writing forcfield type:\n");
	char *forcefield_name="no forcefield";

	if (tng_forcefield_name_set(sink_traj, forcefield_name) != TNG_SUCCESS) fail(&sink_traj,__LINE__);
	printf("Name of force field: %s\n", forcefield_name);

	printf("Setting output file name to %s.\n",filename);
	if (tng_output_file_set(sink_traj, filename) != TNG_SUCCESS) fail(&sink_traj, __LINE__);

	/* start writing molecules */
	create_molecules(&sink_traj, particle_count);
	/* end writing molecules */

	/* start writing box settings */
	printf("Writing box shape\n");
	if (tng_data_block_add(sink_traj, TNG_TRAJ_BOX_SHAPE, "BOX SHAPE", TNG_DOUBLE_DATA, TNG_NON_TRAJECTORY_BLOCK, 1, dimensions, 1, TNG_UNCOMPRESSED, box) == TNG_CRITICAL) fail(&sink_traj, 7);
	/* end writing box settings */

	/* start writing comments */
	printf("Adding annotation block.\n");
	char *annotation = "This is just a test file - not a real simulation trajectory.";
	if (tng_data_block_add(sink_traj, TNG_TRAJ_GENERAL_COMMENTS, "COMMENTS", TNG_CHAR_DATA, TNG_NON_TRAJECTORY_BLOCK, 1, 1, 1, TNG_UNCOMPRESSED, annotation) != TNG_SUCCESS) fail(&sink_traj, 8);
	/* end writing comments */

	if(tng_time_per_frame_set(sink_traj,dt) != TNG_SUCCESS) fail(&sink_traj, __LINE__); // TODO: this has no effect

	printf("Writing file headers (including non-trajectory-data blocks).\n");
	if (tng_file_headers_write(sink_traj, hash_mode) == TNG_CRITICAL) fail(&sink_traj, __LINE__);

	int64_t frames_per_frame_set = 0;
	if (tng_num_frames_per_frame_set_get(sink_traj, &frames_per_frame_set) != TNG_SUCCESS) fail(&sink_traj,__LINE__);
	printf("Preparing to write %d frames per frame set", frames_per_frame_set);

	int frame_data_size = sizeof(double) * particle_count * dimensions;

	printf(".\n  Allocating memory for data block");
	double *data = malloc(frame_data_size * frames_per_frame_set);
	if (!data) fail(&sink_traj, 10);

	printf(".\n  Allocating memory for positions");
	double *molecule_pos = malloc(frame_data_size);
	if (!molecule_pos) fail(&sink_traj, 11);

	printf(".\n  Allocating memory for accelerations");
	double *molecule_acc = malloc(frame_data_size);
	if (!molecule_acc) fail(&sink_traj, 12);

	printf(".\n  Allocating memory for forces");
	double *molecule_frc = malloc(frame_data_size);
	if (!molecule_frc) fail(&sink_traj, 13);

	printf(".\n  Allocating memory for velocities");
	double *molecule_vel = malloc(frame_data_size);
	if (!molecule_vel) fail(&sink_traj, 14);

	initialize(particle_count, dimensions, box, molecule_pos, molecule_vel, molecule_acc);

	double mass = 2.0;
	double potential, kinetic;

	printf(".\n  Generating data\n");

	int step = 0;
	int frame_number = 0;
	int frame_set_number = 0;

	int particle_number = 0;
	int dimension = 0;
	int index = 0;

	int verbosity = 1;
	printf("Running %d steps without storing data.\n",initial_offset);
	if (initial_offset>0){
		while (step < initial_offset) {
			compute(particle_count, dimensions, molecule_pos, molecule_vel, mass, molecule_frc, &potential, &kinetic);
			update(particle_count, dimensions, box, molecule_pos, molecule_vel, molecule_frc, molecule_acc, mass, dt);
			step++;
		}
	}
	step=0;

	printf("start_time=");
	print_time();
	printf("\n");

	while (step < step_count) {

		compute(particle_count, dimensions, molecule_pos, molecule_vel, mass, molecule_frc, &potential, &kinetic);
		update(particle_count, dimensions, box, molecule_pos, molecule_vel, molecule_frc, molecule_acc, mass, dt);

		if ((save>0) && (step % step_save == 0)) {
			if (verbosity > 1) printf("step %d used as frame %d\n", step, frame_number);
			for (particle_number = 0; particle_number < particle_count; particle_number++) {

				// set particle position in each dimension
				for (dimension = 0; dimension < dimensions; dimension++) {
					data[index++] = molecule_pos[particle_number * dimensions + dimension];
				}
			}
			frame_number++;

			if ((frame_number % frames_per_frame_set == 0)) {
				/* Frame set full. Write block and go on */

				if (verbosity > 0)
					printf("writing block %d\n", frame_set_number);

				// create new frameset for data
				if (tng_frame_set_new(sink_traj, frame_set_number * frames_per_frame_set, frames_per_frame_set) != TNG_SUCCESS) {
					free_structures(data, molecule_vel, molecule_frc, molecule_acc, molecule_pos);
					fail(&sink_traj, 15);
				}

				// add data to trajectory
				if (tng_particle_data_block_add(sink_traj, TNG_TRAJ_POSITIONS, "POSITIONS", TNG_DOUBLE_DATA, TNG_TRAJECTORY_BLOCK, frames_per_frame_set, dimensions, 1, 0, particle_count, codec_id, data) != TNG_SUCCESS) {
					free_structures(data, molecule_vel, molecule_frc, molecule_acc, molecule_pos);
					fail(&sink_traj, 16);
				}

				if (tng_frame_set_write(sink_traj, hash_mode) != TNG_SUCCESS) { // write the frameset including data to file
					free_structures(data, molecule_vel, molecule_frc, molecule_acc, molecule_pos);
					fail(&sink_traj, 17);
				}
				frame_set_number++;
				index = 0;
			}

		} else {
			if (verbosity > 2) printf("step %d\n", step);
		}
		if (verbosity > 3) {
			printf("%f", molecule_pos[0]);
			for (dimension = 1; dimension < dimensions; dimension++) {
				printf(" / %f", molecule_pos[dimension]);
			}
			printf("\n");
		}

		step++;
	}

	free_structures(data, molecule_vel, molecule_frc, molecule_acc, molecule_pos);
	tng_trajectory_destroy(&sink_traj); // finalizing and closing trajectory file, free memory

	printf("end_time=");
	print_time();
	printf("\n");

	return filename;
}

/* copy molecule data from source trajectory to destination trajectory */
int transfer_molecules(tng_trajectory_t src_traj,tng_trajectory_t sink_traj){
	int64_t number_of_molecule_types;
	if (tng_num_molecule_types_get(src_traj, &number_of_molecule_types)!= TNG_SUCCESS) fail2(&src_traj,&sink_traj, __LINE__);
	printf("There are %d types of molecules:\n",number_of_molecule_types);

	int mol_type_index;
	for (mol_type_index=0; mol_type_index<number_of_molecule_types; mol_type_index++){
		tng_molecule_t src_molecule,sink_molecule;
		if (tng_molecule_of_index_get(src_traj,mol_type_index,&src_molecule) != TNG_SUCCESS) fail2(&src_traj,&sink_traj,__LINE__);

		char molecule_name[TNG_MAX_STR_LEN];
		if (tng_molecule_name_get(src_traj, src_molecule, molecule_name, TNG_MAX_STR_LEN) != TNG_SUCCESS) fail2(&src_traj,&sink_traj,__LINE__);
		printf("-molecule type %d: %s\n",mol_type_index,molecule_name);

		if (tng_molecule_add(sink_traj, molecule_name, &sink_molecule) != TNG_SUCCESS) fail2(&src_traj,&sink_traj,__LINE__);

		int64_t chain_count;
		if(tng_molecule_num_chains_get(src_traj,src_molecule,&chain_count) != TNG_SUCCESS) fail2(&src_traj,&sink_traj,__LINE__);
		printf(" this molecule has %d chains:\n",chain_count);

		int chain_index;
		for (chain_index=0; chain_index<chain_count; chain_index++){

			tng_chain_t src_chain,sink_chain;
			if(tng_molecule_chain_of_index_get(src_traj,src_molecule,chain_index,&src_chain) != TNG_SUCCESS) fail2(&src_traj,&sink_traj,__LINE__);

			char chain_name[TNG_MAX_STR_LEN];
			if(tng_chain_name_get(src_traj,src_chain,chain_name,TNG_MAX_STR_LEN) != TNG_SUCCESS) fail2(&src_traj,&sink_traj,__LINE__);
			printf(" -chain %d: %s\n",chain_index,chain_name);

			if(tng_molecule_chain_add(sink_traj, sink_molecule, chain_name, &sink_chain) != TNG_SUCCESS) fail2(&src_traj,&sink_traj,__LINE__);

			int64_t residue_count;
			if(tng_chain_num_residues_get(src_traj,src_chain,&residue_count) != TNG_SUCCESS) fail2(&src_traj,&sink_traj,__LINE__);
			printf("  has %d residues:\n",residue_count);

			int residue_index;
			for (residue_index=0; residue_index<residue_count; residue_index++){

				tng_residue_t src_residue,sink_residue;
				if(tng_chain_residue_of_index_get(src_traj,src_chain,residue_index,&src_residue) != TNG_SUCCESS) fail2(&src_traj,&sink_traj,__LINE__);

				char residue_name[TNG_MAX_STR_LEN];
				if(tng_residue_name_get(src_traj,src_residue,residue_name,TNG_MAX_STR_LEN) != TNG_SUCCESS) fail2(&src_traj,&sink_traj,__LINE__);
				printf("  -residue %d: %s\n",residue_index,residue_name);

				if (tng_chain_residue_add(sink_traj, sink_chain, residue_name, &sink_residue) != TNG_SUCCESS) fail2(&src_traj,&sink_traj,__LINE__);

				int64_t atom_count;
				if(tng_residue_num_atoms_get(src_traj,src_residue,&atom_count) != TNG_SUCCESS) fail2(&src_traj,&sink_traj,__LINE__);
				printf("   has %d atoms:\n",atom_count);

				int atom_index;
				for (atom_index=0; atom_index<atom_count; atom_index++){

					tng_atom_t src_atom,sink_atom;
					if (tng_residue_atom_of_index_get(src_traj,src_residue,atom_index,&src_atom) != TNG_SUCCESS) fail2(&src_traj,&sink_traj,__LINE__);

					char atom_name[TNG_MAX_STR_LEN];
					if (tng_atom_name_get(src_traj,src_atom,atom_name,TNG_MAX_STR_LEN) != TNG_SUCCESS) fail2(&src_traj,&sink_traj,__LINE__);
					char atom_type[TNG_MAX_STR_LEN];
					if (tng_atom_type_get(src_traj,src_atom,atom_type,TNG_MAX_STR_LEN) != TNG_SUCCESS) fail2(&src_traj,&sink_traj,__LINE__);
					printf("   -atom %d is of type %s: %s\n",atom_index,atom_type,atom_name);

					if(tng_residue_atom_add(sink_traj, sink_residue, atom_name, atom_type, &sink_atom) != TNG_SUCCESS) fail2(&src_traj,&sink_traj,__LINE__);
			}
		}
	}
	int64_t molecule_count;
	if(tng_molecule_cnt_get(src_traj,src_molecule,&molecule_count) != TNG_SUCCESS) fail2(&src_traj,&sink_traj,__LINE__);
	if(tng_molecule_cnt_set(sink_traj, sink_molecule, molecule_count) != TNG_SUCCESS) fail2(&src_traj,&sink_traj,__LINE__);
		printf(" set number of %s molecules to %d\n",molecule_name,molecule_count);
	}
}

/* copy box data from source trajectory to destination trajectory */
int transfer_box_data(tng_trajectory_t src_traj,tng_trajectory_t sink_traj, double **box, int64_t *dimensions, char *type){
	int64_t dimension,number_of_box_shape_frames;
	union data_values **box_data = 0;

	if (tng_data_get(src_traj,TNG_TRAJ_BOX_SHAPE,&box_data,&number_of_box_shape_frames,dimensions,type)!=TNG_SUCCESS)fail2(&src_traj,&sink_traj,__LINE__);
	printf("Read %d box dimensions: ",*dimensions);
	int size=*dimensions * sizeof(double);
	*box=malloc(*dimensions * sizeof(double));
	for (dimension=0; dimension<*dimensions;dimension++){
		if (*type == TNG_DOUBLE_DATA){
			(*box)[dimension] = box_data[0][dimension].d;
			printf("%f",(*box)[dimension]);
			if (dimension+1<*dimensions){
				printf(" x ");
			}
		} else {
			fail2(&src_traj,&sink_traj,__LINE__);
		}
	}
	printf(".\n");

	if (sink_traj != NULL){
		if (tng_data_block_add(sink_traj, TNG_TRAJ_BOX_SHAPE, "BOX SHAPE", *type, TNG_NON_TRAJECTORY_BLOCK, 1, *dimensions, 1, TNG_UNCOMPRESSED, *box) != TNG_SUCCESS) fail2(&src_traj,&sink_traj, 472);
		printf("Box size written to destination trajectory.\n");
	}
}

/* copy comment data from source trajectory to destination trajectory */
int transfer_comments(tng_trajectory_t src_traj,tng_trajectory_t sink_traj){
	int64_t num_comments,comment_frame_num;
	char comment_type;
	union data_values **comment_data = 0;
	if(tng_data_get(src_traj,TNG_TRAJ_GENERAL_COMMENTS,&comment_data,&comment_frame_num,&num_comments,&comment_type) != TNG_SUCCESS) fail2(&src_traj,&sink_traj,__LINE__);
	if (comment_type != TNG_CHAR_DATA)fail2(&src_traj,&sink_traj,__LINE__);
	char *annotation=0;
	switch (num_comments){
		case 1:
			annotation = comment_data[0][0].c;
			printf("Found comment: \"%s\"\n",annotation);
			printf("Adding comment block.\n");
			if (tng_data_block_add(sink_traj, TNG_TRAJ_GENERAL_COMMENTS, "COMMENTS", TNG_CHAR_DATA, TNG_NON_TRAJECTORY_BLOCK, 1, 1, 1, TNG_UNCOMPRESSED, annotation) != TNG_SUCCESS) fail2(&src_traj,&sink_traj,__LINE__);
			break;
		case 0:
			break;
		default:
			fail2(&src_traj,&sink_traj,__LINE__);
	}
}


int rewrite_file(char *sink_tng_file, int64_t dst_frames_per_block, int steps_save,int64_t sink_codec_id, double precision, char *src_tng_file) {
	/* Start initialization */
	printf("Preparing to read stored trajectory...\n");
	tng_trajectory_t src_traj; // source trajectory handle
	if (tng_trajectory_init(&src_traj) != TNG_SUCCESS) fail(&src_traj, __LINE__);

	char hash_mode = TNG_USE_HASH;

	printf("Setting input file to %s.\n", src_tng_file);
	if (tng_input_file_set(src_traj, src_tng_file) != TNG_SUCCESS) fail(&src_traj, __LINE__);
	if (tng_file_headers_read(src_traj, hash_mode) != TNG_SUCCESS) fail(&src_traj, __LINE__);

	printf("Initializing trajectory storage:\n");
	tng_trajectory_t sink_traj; // sink trajectory handle
	if (tng_trajectory_init(&sink_traj) != TNG_SUCCESS) fail2(&src_traj,&sink_traj, __LINE__);

	int64_t medium_stride_length;

	if (tng_medium_stride_length_get(src_traj, &medium_stride_length) != TNG_SUCCESS) fail2(&src_traj,&sink_traj, __LINE__);
	if (tng_medium_stride_length_set(sink_traj, medium_stride_length) != TNG_SUCCESS) fail2(&src_traj,&sink_traj, __LINE__);
	printf("medium stride length: %d\n", medium_stride_length);

	int64_t long_stride_length;

	if (tng_long_stride_length_get(src_traj, &long_stride_length) != TNG_SUCCESS) fail2(&src_traj,&sink_traj, __LINE__);
	if (tng_long_stride_length_set(sink_traj, long_stride_length) != TNG_SUCCESS) fail2(&src_traj,&sink_traj, __LINE__);
	printf("long stride length: %d\n", long_stride_length);

	int64_t src_frames_per_block;
    if (tng_num_frames_per_frame_set_get(src_traj,&src_frames_per_block) != TNG_SUCCESS) fail2(&src_traj,&sink_traj, __LINE__);
	if (tng_num_frames_per_frame_set_set(sink_traj,dst_frames_per_block) != TNG_SUCCESS) fail2(&src_traj,&sink_traj, __LINE__);
	printf("Number of frames per block (source => sink): %d => %d.\n",src_frames_per_block,dst_frames_per_block);

	printf("reading computer information:\n");


	char username[TNG_MAX_STR_LEN];
	if (tng_first_user_name_get(src_traj, username, TNG_MAX_STR_LEN) != TNG_SUCCESS) fail2(&src_traj,&sink_traj, __LINE__);
	if (tng_first_user_name_set(sink_traj, username) != TNG_SUCCESS) fail2(&src_traj,&sink_traj, __LINE__);
	printf("File was created by %s", username);

	char prog_name[TNG_MAX_STR_LEN];
	if (tng_first_program_name_get(src_traj, prog_name, TNG_MAX_STR_LEN) != TNG_SUCCESS) fail2(&src_traj,&sink_traj, __LINE__);

	if (tng_first_program_name_set(sink_traj, prog_name) != TNG_SUCCESS) fail2(&src_traj,&sink_traj, __LINE__);
	printf(" running %s", prog_name);

	char computer_name[TNG_MAX_STR_LEN];
	if(tng_first_computer_name_get(src_traj, computer_name, TNG_MAX_STR_LEN) != TNG_SUCCESS) fail2(&src_traj,&sink_traj, __LINE__);


	if (tng_first_computer_name_set(sink_traj, computer_name) != TNG_SUCCESS) fail2(&src_traj,&sink_traj, __LINE__);
	printf(" on %s\n", computer_name);

	printf("writing forcfield type:\n");
	char forcefield_name[TNG_MAX_STR_LEN];
	if(tng_forcefield_name_get(src_traj, forcefield_name, TNG_MAX_STR_LEN) != TNG_SUCCESS) fail2(&src_traj,&sink_traj, __LINE__);

	if (tng_forcefield_name_set(sink_traj, forcefield_name) != TNG_SUCCESS) fail2(&src_traj,&sink_traj, __LINE__);
	printf("Name of force field: %s\n", forcefield_name);

	printf("Setting output file name to %s.\n", sink_tng_file);
	if (tng_output_file_set(sink_traj, sink_tng_file) != TNG_SUCCESS) fail2(&src_traj,&sink_traj, __LINE__);

	/* start writing molecules */
	transfer_molecules(src_traj,sink_traj);
	/* end writing molecules */

	/* start writing box settings */
	printf("Writing box shape\n");
	int64_t dimensions;
	char type;
	double *box=NULL;
	transfer_box_data(src_traj,sink_traj,&box,&dimensions,&type);
	/* end writing box settings */

	/* start writing comments */
	printf("Adding annotation block.\n");
	transfer_comments(src_traj,sink_traj);
	/* end writing comments */

	double time_per_frame;
	if(tng_time_per_frame_get(src_traj, &time_per_frame) != TNG_SUCCESS) fail2(&src_traj,&sink_traj,__LINE__);
	printf("time per frame: %f\n", time_per_frame);
	// TODO: somehow this returns a wrong value
	if(tng_time_per_frame_set(sink_traj,0.0002) != TNG_SUCCESS) fail2(&src_traj,&sink_traj,__LINE__);

	printf("Writing file headers (including non-trajectory-data blocks).\n");
	if (tng_file_headers_write(sink_traj, hash_mode) != TNG_SUCCESS) fail2(&src_traj,&sink_traj,494);

	int64_t frames_per_frame_set = 0;
	if (tng_num_frames_per_frame_set_get(sink_traj, &frames_per_frame_set) != TNG_SUCCESS) fail2(&src_traj,&sink_traj,__LINE__);
	printf("Preparing to write %d frames per frame set", frames_per_frame_set);




	if (tng_compression_precision_set(sink_traj, 1/precision) != TNG_SUCCESS) fail2(&src_traj,&sink_traj, __LINE__);
	printf("Set compression precision: %f\n", precision);
	/* End of Initialization */







	int64_t stride_length;
	if (tng_data_get_stride_length(src_traj, TNG_TRAJ_POSITIONS, 0, &stride_length)!=TNG_SUCCESS) fail2(&src_traj,&sink_traj,__LINE__);
	printf("stride length of first frame is %d.\n", stride_length);

	int64_t source_number_of_blocks;
    if(tng_num_frame_sets_get(src_traj, &source_number_of_blocks) != TNG_SUCCESS) fail2(&src_traj,&sink_traj,__LINE__);


    printf("total number of frame sets in source file: %d.\n",source_number_of_blocks);

	transfer_box_data(sink_traj,NULL,&box,&dimensions,&type);


	union data_values ***src_positions = 0; // data structure to handle position data between input and output
	double *dst_positions=NULL;
	int dst_pos_index=0;
    int64_t src_block_num=0;
    int64_t src_frame_num=0;

    int64_t dst_frame_num=0;

    /* loop through each block of the source file */




    while (src_block_num<source_number_of_blocks){
		if (tng_frame_set_read(src_traj,hash_mode) != TNG_SUCCESS) fail2(&src_traj,&sink_traj,__LINE__);

		int64_t src_number_of_particles,src_values_per_particle,src_number_of_frames_in_current_block;
		if (tng_particle_data_get(src_traj,TNG_TRAJ_POSITIONS,&src_positions,&src_number_of_frames_in_current_block,&src_number_of_particles,&src_values_per_particle,&type) != TNG_SUCCESS) fail2(&src_traj,&sink_traj,__LINE__);
		printf("Opened frameset %d (%d frames with %d particles, each having %d values).\n",src_block_num,src_number_of_frames_in_current_block,src_number_of_particles,src_values_per_particle);

		if (src_values_per_particle != dimensions){
			printf("This does not match the number of dimensions! Aborting now.\n");
			fail2(&src_traj,&sink_traj,__LINE__);
		}
		if (type != TNG_DOUBLE_DATA) fail2(&src_traj,&sink_traj,__LINE__);

		char block_name[TNG_MAX_STR_LEN];
		if(tng_data_block_name_get(src_traj, TNG_TRAJ_POSITIONS, block_name, TNG_MAX_STR_LEN) != TNG_SUCCESS) fail2(&src_traj,&sink_traj,__LINE__);

		int64_t src_values_per_frame=src_number_of_particles*src_values_per_particle;

		if (src_frame_num == 0) {
			printf("allocating space for dst_postisions: ");
			dst_positions=malloc(dst_frames_per_block * src_values_per_frame * sizeof(double));
			printf("done\n");
		}

		int64_t src_local_frame;
		/* loop through all frames of the current block */
		for (src_local_frame=0; src_local_frame<src_number_of_frames_in_current_block; src_local_frame++){
			printf("read frame %d",src_frame_num);
			/* resampling option: only store frames that meet the new sampling rate */
			if (src_frame_num % steps_save == 0){

				int particle;
				/* get positions for all particles */
				for (particle=0; particle<src_number_of_particles;particle++){
					int dimension;

					/* for each particle get location in each dimension */
					for (dimension=0; dimension<dimensions; dimension++){
						double value=src_positions[src_local_frame][particle][dimension].d;
						dst_positions[dst_pos_index]=value;
						dst_pos_index++;
					}
				} // fore each particle
				dst_frame_num++;
				printf(" - will be stored!\n");

				if (dst_frame_num % dst_frames_per_block == 0){
					printf("writing block.\n");
					if (tng_frame_set_new(sink_traj, dst_frame_num-dst_frames_per_block, dst_frames_per_block)!= TNG_SUCCESS){
						fail2(&src_traj,&sink_traj,__LINE__);
					}
					printf("  created new frame set\n");

					if (tng_particle_data_block_add(sink_traj, TNG_TRAJ_POSITIONS, block_name, type, TNG_TRAJECTORY_BLOCK, dst_frames_per_block, dimensions, stride_length, 0, src_number_of_particles, sink_codec_id, dst_positions) != TNG_SUCCESS){
						fail2(&src_traj,&sink_traj,__LINE__);
					}
					printf("  added particle data block\n");

					if (tng_frame_set_write(sink_traj, hash_mode) != TNG_SUCCESS){
						fail2(&src_traj,&sink_traj,__LINE__);
					}
					printf("  written frame set\n");

					dst_pos_index=0;
					printf("flushed.\n");
				}


			} // if frame meets resampling rate
			else {
				printf("\n");
			}
			src_frame_num++;
		}

    	src_block_num++;
    }

    free(dst_positions);
	free(box);
	tng_trajectory_destroy(&sink_traj);
	tng_trajectory_destroy(&src_traj);

	return 0;
}

int translate_file(char *src_tng_file){
	/* Start initialization */
	printf("Preparing to read stored trajectory...\n");
	tng_trajectory_t src_traj; // source trajectory handle
	if (tng_trajectory_init(&src_traj) != TNG_SUCCESS) fail(&src_traj, __LINE__);

	printf("Setting input file to %s.\n", src_tng_file);
	if (tng_input_file_set(src_traj, src_tng_file) != TNG_SUCCESS) fail(&src_traj, __LINE__);

	char hash_mode = TNG_USE_HASH;
	if (tng_file_headers_read(src_traj, hash_mode) != TNG_SUCCESS) fail(&src_traj, __LINE__);
	/* End of Initialization */

	/* get box data (we actually need type and dimensions) */
	int64_t dimensions;
	char type;
	double *box=NULL;
	transfer_box_data(src_traj,NULL,&box,&dimensions,&type);
	/* end get box data */

	int64_t source_number_of_blocks;
    if(tng_num_frame_sets_get(src_traj, &source_number_of_blocks) != TNG_SUCCESS) fail(&src_traj,__LINE__);
    printf("total number of frame sets in file: %d.\n",source_number_of_blocks);

    FILE *lammpfile;
    {	// open lammpstrj file for writing trajectory data in vmd-readable format
        char *lammp_output_file;
    	char *extension = ".lammpstrj";
    	lammp_output_file=malloc(strlen(src_tng_file)+strlen(extension)+1);
    	lammp_output_file[0]='\0';
    	strcat(lammp_output_file,src_tng_file);
    	strcat(lammp_output_file,extension);
    	lammpfile = fopen(lammp_output_file,"w");
        free(lammp_output_file);
        printf("Opened %s for writing.\n",lammp_output_file);
    }

	union data_values ***src_positions = 0; // data structure to handle position data between input and output
	double *positions=NULL;
    int64_t src_block_num=0;
    int64_t src_frame_num=0;

    /* loop through each block of the source file */
    while (src_block_num<source_number_of_blocks){
		if (tng_frame_set_read(src_traj,hash_mode) != TNG_SUCCESS) fail(&src_traj,__LINE__);

		int64_t src_number_of_particles,src_values_per_particle,src_number_of_frames_in_current_block;
		if (tng_particle_data_get(src_traj,TNG_TRAJ_POSITIONS,&src_positions,&src_number_of_frames_in_current_block,&src_number_of_particles,&src_values_per_particle,&type) != TNG_SUCCESS) fail(&src_traj,__LINE__);
		printf("Opened frameset %d (%d frames with %d particles, each having %d values).\n",src_block_num,src_number_of_frames_in_current_block,src_number_of_particles,src_values_per_particle);

		if (src_values_per_particle != dimensions){
			printf("This does not match the number of dimensions! Aborting now.\n");
			fail(&src_traj,__LINE__);
		}
		if (type != TNG_DOUBLE_DATA) fail(&src_traj,__LINE__);

		char block_name[TNG_MAX_STR_LEN];
		if(tng_data_block_name_get(src_traj, TNG_TRAJ_POSITIONS, block_name, TNG_MAX_STR_LEN) != TNG_SUCCESS) fail(&src_traj,__LINE__);

		int64_t src_values_per_frame=src_number_of_particles*src_values_per_particle;

		positions=malloc(src_number_of_frames_in_current_block * src_values_per_frame * sizeof(double));

		int64_t src_local_frame;
		/* loop through all frames of the current block */
		for (src_local_frame=0; src_local_frame<src_number_of_frames_in_current_block; src_local_frame++){
			int index=0;
			/* resampling option: only store frames that meet the new sampling rate */

			int particle;
			/* get positions for all particles */
			for (particle=0; particle<src_number_of_particles;particle++){
				int dimension;
				/* for each particle get location in each dimension */
				for (dimension=0; dimension<dimensions; dimension++){
					double value=src_positions[src_local_frame][particle][dimension].d;
					positions[index]=value;
					index++;
				}
			} // fore each particle
			lammpstrj_write_positions(lammpfile,src_frame_num,dimensions,box,src_number_of_particles,positions);
			printf("read frame %d.\n",src_frame_num);
			src_frame_num++;
		}
	    free(positions);
    	src_block_num++;
    }

	free(box);
	tng_trajectory_destroy(&src_traj);
	fclose(lammpfile); // close lammp file

	return 0;

}

/******************************************************************************/

/* arguments taken:
 * <number of blocks> <number of frames per block> <cubic box length> <filename>
 */
int generate(char *orig_file,int frames_per_block,int step_save,int64_t codec_id, double precision, int framesets,int particle_count,int box_size,int initial_offset,int save){

	int step_count = framesets*frames_per_block*step_save;
	int dimensions = 3;
	double dt = 0.0002;
	int proc_count = omp_get_num_procs();
	int thread_count = omp_get_max_threads();
	double box[dimensions];
	int dim = 0;
	for (; dim < dimensions; dim++) {
		box[dim] = box_size+dim;
	}
	printf("Simulating %dx%dx%d box with %d molecules. ",box_size,box_size+1,box_size+2,particle_count);
	if (step_save<10 || step_save>15){
		switch (step_save%10){
			case 1:
				printf(" Storing every %dst frame.",step_save);
				break;
			case 2:
				printf(" Storing every %dnd frame.",step_save);
				break;
			case 3:
				printf(" Storing every %drd frame.",step_save);
				break;
			default:
				printf(" Storing every %dth frame.",step_save);
				break;
		}
	} else {
		printf(" Storing every %dth frame.",step_save);
	}
	printf(" Generating %d frame sets with each %d frames.\n",framesets,frames_per_block);
	printf("  Number of processors available = %d\n", proc_count);
	printf("  Number of threads =              %d\n", thread_count);

	orig_file = create_trajectory(particle_count, step_count, initial_offset, step_save, dt, proc_count, thread_count, dimensions, box,frames_per_block,orig_file,save,codec_id, precision);
	printf("\nWritten original trajectory to %s.\n",orig_file);
	return 0;
}


int show_help(char *argv[]){
	printf("Usage:\n");
	printf("%s -g [-b <blocks>] [-c <codec>] [-d <precision>] [-f <fpb>] [-o <offset>] [-p <parts>] [-s <save_interval>] [-x <box_size>] -n <outfile>\n\tto generate a trajectory or\n",*argv);
	printf("%s  -r  -i <infile> [-c <codec>] [-d <precision>] [-f <fpb>] [-s <save_interval>] -n <outfile> \n\tto rewrite (compress) a trajectory or\n",*argv);
	printf("%s -t <infile>\n\tto translate the tng file to a lammpstrj file or\n",*argv);
	printf("%s -v\nto print the version of the used hrtc lib.");
	printf("\n");
	printf("<blocks>       : number of frame sets (blocks)\n");
	printf("<box_size>     : length of the reactor box\n");
	printf("<codec>        = NONE | TNG | HRTC\n");
	printf("<fpb>          : number of frames per block/frame set\n");
	printf("<infile>       : name of the input file\n");
	printf("<offset>       : number of frames to simulate BEFORE storagestarts.\n");
	printf("<outfile>      : name of the output file\n");
	printf("<parts>        : number of particles\n");
	printf("<precision>    : precision value for compression (default: 0.001)\n");
	printf("<save_interval>: determines, which frames of the simulation shall be saved.\n");
	printf("\n");
}

void print_version(){
	hrtc_version();
}

int main(int argc, char *argv[]) {
	setlinebuf(stdout);

	int cmd_arg;

	int generate_flag=0;
	int rewrite_flag=0;
	int translate_flag=0;
	int number_of_blocks=10;
	int number_of_particles=32;
	int frames_per_block=100;
	int box_size=10;
	int codec=TNG_UNCOMPRESSED;
	int save=1;
	int steps_save=100;
	int initial_offset=0;
	double precision=0.001;
	char *in_filename=NULL;
	char *out_filename=NULL;

	while ((cmd_arg=getopt(argc,argv,"b:c:d:f:ghi:n:o:p:rs:t:vx:")) != -1){
		switch (cmd_arg){
			case 'b':
				number_of_blocks=atoi(optarg);
				break;
			case 'c':
				if (strcmp(optarg,"HRTC")==0){
					codec=TNG_HRTC_COMPRESSION;
				}
				if (strcmp(optarg,"NONE")==0){
					codec=TNG_UNCOMPRESSED;
				}
				if (strcmp(optarg,"TNG")==0){
					codec=TNG_TNG_COMPRESSION;
				}
				if (strcmp(optarg,"NO_SAVE")==0){
					save=0;
				}
				break;
			case 'd':
				precision=atof(optarg);
				break;
			case 'f':
				frames_per_block=atoi(optarg);
				break;
			case 'g':
				generate_flag=1;
				break;
			case 'h':
				show_help(argv);
				break;
			case 'i':
				in_filename=optarg;
				break;
			case 'n':
				out_filename=optarg;
				break;
			case 'o':
				initial_offset=atoi(optarg);
				break;
			case 'p':
				number_of_particles=atoi(optarg);
				break;
			case 'r':
				rewrite_flag=1;
				break;
			case 's':
				steps_save=atoi(optarg);
				break;
			case 't':
				translate_flag=1;
				in_filename=optarg;
				break;
			case 'v':
				print_version();
				return 0;
				break;
			case 'x':
				box_size=atoi(optarg);
				break;
			case '?':
				fprintf(stderr,"Option -%c requires an argument!\n",optopt);
				break;
		}

	}

	if (generate_flag + rewrite_flag + translate_flag > 1){
			fprintf(stderr,"-g, -r, and -t are exclusive flags!\n");
			return __LINE__;
	}
	if (! (generate_flag  || rewrite_flag || translate_flag)){
		fprintf(stderr,"You need to specify either -g, -r, or -t!\n");
		return __LINE__;
	}
	if (generate_flag || rewrite_flag){
		if (out_filename==NULL){
			fprintf(stderr,"No output filename given!\n");
			return __LINE__;
		}
	}
	if (rewrite_flag || translate_flag){
		if (in_filename==NULL){
				fprintf(stderr,"No input filename given!\n");
				return __LINE__;
		}
	}

	if (generate_flag)  return generate  (  out_filename, frames_per_block, steps_save, codec, precision, number_of_blocks, number_of_particles, box_size, initial_offset, save);
	if (rewrite_flag)   return rewrite_file(out_filename, frames_per_block, steps_save, codec, precision, in_filename);
	if (translate_flag) return translate_file(in_filename);

	return __LINE__;
}
/******************************************************************************/


void compute(int number_of_particles, int dimensions, double positions[], double vel[], double mass, double forces[], double *pot, double *kin)
/******************************************************************************/
/*
 Purpose:

 COMPUTE computes the forces and energies.

 Discussion:

 The computation of forces and energies is fully parallel.

 The potential function V(X) is a harmonic well which smoothly
 saturates to a maximum value at PI/2:

 v(x) = ( sin ( min ( x, PI2 ) ) )**2

 The derivative of the potential is:

 dv(x) = 2.0 * sin ( min ( x, PI2 ) ) * cos ( min ( x, PI2 ) )
 = sin ( 2.0 * min ( x, PI2 ) )

 Licensing:

 This code is distributed under the GNU LGPL license.

 Modified:

 21 November 2007

 Author:

 Original FORTRAN77 version by Bill Magro.
 C version by John Burkardt.

 Parameters:

 Input, int NP, the number of particles.

 Input, int ND, the number of spatial dimensions.

 Input, double POS[ND*NP], the position of each particle.

 Input, double VEL[ND*NP], the velocity of each particle.

 Input, double MASS, the mass of each particle.

 Output, double F[ND*NP], the forces.

 Output, double *POT, the total potential energy.

 Output, double *KIN, the total kinetic energy.
 */
{
	double distance;
	double d2;
	int dimension;
	int particleB;
	int particleA;
	double ke = 0.0;
	double pe = 0.0;
	double half_Pi = 3.141592653589793 / 2.0;
	double rij[dimensions];

# pragma omp parallel \
    shared ( forces, dimensions, number_of_particles, positions, vel ) \
    private ( dimension, particleB, particleA, rij, distance, d2 )

# pragma omp for reduction ( + : pe, ke )
	for (particleA = 0; particleA < number_of_particles; particleA++) {
		/*
		 Compute the potential energy and forces.
		 */
		for (dimension = 0; dimension < dimensions; dimension++) {
			forces[dimension + particleA * dimensions] = 0.0;
		}

		for (particleB = 0; particleB < number_of_particles; particleB++) {
			if (particleA != particleB) {
				distance = dist(dimensions, positions + particleA * dimensions, positions + particleB * dimensions, rij);
				if (distance < half_Pi) {
					d2 = distance;
				} else {
					d2 = half_Pi;
				}

				pe = pe + 0.5 * pow(sin(d2), 2);
				/*
				 Attribute half of the potential energy to particle J.
				 */
				for (dimension = 0; dimension < dimensions; dimension++) {
					forces[dimension + particleA * dimensions] = forces[dimension + particleA * dimensions] - rij[dimension] * sin(2.0 * d2) / distance;
				}
			}
		}
		/*
		 Compute the kinetic energy.
		 */
		for (dimension = 0; dimension < dimensions; dimension++) {
			ke = ke + vel[dimension + particleA * dimensions] * vel[dimension + particleA * dimensions];
		}
	}

	ke = ke * 0.5 * mass;

	*pot = pe;
	*kin = ke;

	return;
}
/******************************************************************************/

double dist(int dimensions, double r1[], double r2[], double dr[])

/******************************************************************************/
/*
 Purpose:

 DIST computes the displacement (and its norm) between two particles.

 Licensing:

 This code is distributed under the GNU LGPL license.

 Modified:

 21 November 2007

 Author:

 Original FORTRAN77 version by Bill Magro.
 C version by John Burkardt.

 Parameters:

 Input, int ND, the number of spatial dimensions.

 Input, double R1[ND], R2[ND], the positions of the particles.

 Output, double DR[ND], the displacement vector.

 Output, double D, the Euclidean norm of the displacement.
 */
{
	double d;
	int i;

	d = 0.0;
	for (i = 0; i < dimensions; i++) {
		dr[i] = r1[i] - r2[i];
		d = d + dr[i] * dr[i];
	}
	d = sqrt(d);

	return d;
}
/******************************************************************************/

void initialize(int particle_count, int num_dims, double box[], double pos[], double vel[], double acc[]) {
	printf(".\n  Initializing particle positions");
	int curernt_dim;
	int particle;
	int index;
	int seed = 0;
	for (particle = 0; particle < particle_count; particle++) {
		for (curernt_dim = 0; curernt_dim < num_dims; curernt_dim++) {
			index = particle * num_dims + curernt_dim;
			pos[index] = random_value(box[curernt_dim]);
			vel[index] = 0.0;
			acc[index] = 0.0;
		}
	}

	return;
}
/******************************************************************************/

void timestamp(void)

/******************************************************************************/
/*
 Purpose:

 TIMESTAMP prints the current YMDHMS date as a time stamp.

 Example:

 31 May 2001 09:45:54 AM

 Licensing:

 This code is distributed under the GNU LGPL license.

 Modified:

 24 September 2003

 Author:

 John Burkardt

 Parameters:

 None
 */
{
# define TIME_SIZE 40

	static char time_buffer[TIME_SIZE];
	const struct tm *tm;
	time_t now;

	now = time(NULL);
	tm = localtime(&now);

	strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);

	printf("%s\n", time_buffer);

	return;
# undef TIME_SIZE
}
/******************************************************************************/

void update(int number_of_particles, int dimensions, double *box, double pos[], double vel[], double foces[], double acc[], double mass, double dt)

/******************************************************************************/
/*
 Purpose:

 UPDATE updates positions, velocities and accelerations.

 Discussion:

 The time integration is fully parallel.

 A velocity Verlet algorithm is used for the updating.

 x(t+dt) = x(t) + v(t) * dt + 0.5 * a(t) * dt * dt
 v(t+dt) = v(t) + 0.5 * ( a(t) + a(t+dt) ) * dt
 a(t+dt) = f(t) / m

 Licensing:

 This code is distributed under the GNU LGPL license.

 Modified:

 17 April 2009

 Author:

 Original FORTRAN77 version by Bill Magro.
 C version by John Burkardt.

 Parameters:

 Input, int NP, the number of particles.

 Input, int ND, the number of spatial dimensions.

 Input/output, double POS[ND*NP], the position of each particle.

 Input/output, double VEL[ND*NP], the velocity of each particle.

 Input, double F[ND*NP], the force on each particle.

 Input/output, double ACC[ND*NP], the acceleration of each particle.

 Input, double MASS, the mass of each particle.

 Input, double DT, the time step.
 */
{
	int dimension;
	int particle_num;
	double rmass;
	int index;

	rmass = 1.0 / mass;

# pragma omp parallel \
    shared ( acc, dt, foces, dimensions, number_of_particles, pos, rmass, vel, box ) \
    private ( dimension, particle_num, index )

# pragma omp for
	for (particle_num = 0; particle_num < number_of_particles; particle_num++) {
		for (dimension = 0; dimension < dimensions; dimension++) {
			index = particle_num * dimensions + dimension;

			pos[index] = pos[index] + vel[index] * dt + 0.5 * acc[index] * dt * dt;
			if (pos[index] > box[dimension]) {
				pos[index] = 2 * box[dimension] - pos[index];
				vel[index] = -vel[index];
			}
			if (pos[index] < 0) {
				pos[index] = -pos[index];
				vel[index] = -vel[index];
			}
			vel[index] = vel[index] + 0.5 * dt * (foces[index] * rmass + acc[index]);
			acc[index] = foces[index] * rmass;
		}
	}

	return;
}

#endif
