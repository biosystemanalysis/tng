#ifndef _TNGIO_HPP
#define _TNGIO_HPP     

#include "tng_io.h"

namespace Tng
{
class Trajectory;
class Atom;
class Residue;
class Chain;
class Molecule;
typedef class Molecule * Molecule_t;


class Trajectory {
private:
	tng_trajectory_t traj;
	tng_function_status status;
public:
		/**
 * @brief Add a molecule to the trajectory.
 * @param name is a pointer to the string containing the name of the new molecule.
 * @param molecule is a pointer to the newly created molecule.
 * @return TNG_SUCCESS (0) if successful or TNG_CRITICAL (2) if a major
 * error has occured.
 */

	tng_function_status addMolecule(const char *, Molecule_t);
	friend class Atom;
	friend class Residue;
	friend class Chain;
	friend class Molecule;
	
	//! Normal constructor
	Trajectory()
	{ status = tng_trajectory_init(&traj); }

	//! Copy constructor
	Trajectory(Trajectory * src)
	{ status = tng_trajectory_init_from_src(traj,&src->traj); }

	//! Detructor
	~Trajectory()
	{ status = tng_trajectory_destroy(&traj); }

	//! Status
	tng_function_status getStatus()
	{ return status; }


	/**
	* @brief Get the name of the input file.
	* @param file_name the string to fill with the name of the input file,
	* memory must be allocated before.
	* @param max_len maximum char length of the string, i.e. how much memory has
	* been reserved for file_name. This includes \0 terminating character.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred (source string longer than destination string).
	*/
	tng_function_status getInputFile (char *file_name, const int max_len)
	{
		return status = tng_input_file_get(traj, file_name,   max_len);
	}

	/**
	* @brief Set the name of the input file.
	* @param file_name the name of the input file.
	* @return TNG_SUCCESS (0) if successful or TNG_CRITICAL (2) if a major
	* error has occured.
	*/
	tng_function_status setInputFile(const char *file_name)
	{
		return status = tng_input_file_set(traj, file_name);
	}


	/**
	* @brief Get the name of the output file.
	* @param file_name the string to fill with the name of the output file,
	* memory must be allocated before.
	* @param max_len maximum char length of the string, i.e. how much memory has
	* been reserved for file_name. This includes \0 terminating character.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred (source string longer than destination string).
	*/
	tng_function_status getOutputFile(char *file_name, const int max_len)
	{
		return status = tng_output_file_get(traj, file_name, max_len);
	}


	/**
	* @brief Set the name of the output file.
	* @param tng_data the trajectory of which to set the output file name.
	* @param file_name the name of the output file.
	* @return TNG_SUCCESS (0) if successful or TNG_CRITICAL (2) if a major
	* error has occured.
	*/
	tng_function_status setOutputFile(const char *file_name)
	{
		return status = tng_output_file_set(traj, file_name);
	}


	/**
	* @brief Get the name of the program used when creating the trajectory.
	* @param name the string to fill with the name of the program,
	* memory must be allocated before.
	* @param max_len maximum char length of the string, i.e. how much memory has
	* been reserved for name. This includes \0 terminating character.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred (source string longer than destination string).
	*/
	tng_function_status getFirstProgramName(char *name, const int max_len)
	{
		return status = tng_first_program_name_get(traj,name,max_len);
	}


	/**
	* @brief Set the name of the program used when creating the trajectory..
	* @param new_name is a string containing the wanted name.
	* @return TNG_SUCCESS (0) if successful or TNG_CRITICAL (2) if a major
	* error has occured.
	*/
	tng_function_status setFirstProgramName(const char *new_name)
	{
		return status = tng_first_program_name_set(traj, new_name);
	}


	/**
	* @brief Get the name of the program used when last modifying the trajectory.
	* @param name the string to fill with the name of the program,
	* memory must be allocated before.
	* @param max_len maximum char length of the string, i.e. how much memory has
	* been reserved for name. This includes \0 terminating character.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred (source string longer than destination string).
	*/
	tng_function_status getLastProgramName(char *name, const int max_len)
	{
		return status = tng_last_program_name_get(traj, name, max_len);
	}


	/**
	* @brief Set the name of the program used when last modifying the trajectory.
	* @param new_name is a string containing the wanted name.
	* @return TNG_SUCCESS (0) if successful or TNG_CRITICAL (2) if a major
	* error has occured.
	*/
	tng_function_status setLastProgramName(const char *new_name)
	{
		return status = tng_last_program_name_set(traj, new_name);
	}


	/**
	* @brief Get the name of the user who created the trajectory.
	* @param name the string to fill with the name of the user,
	* memory must be allocated before.
	* @param max_len maximum char length of the string, i.e. how much memory has
	* been reserved for name. This includes \0 terminating character.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred (source string longer than destination string).
	*/
	tng_function_status getFirstUserName(char *name, const int max_len)
	{
		return status = tng_first_user_name_get(traj,name, max_len);
	}


	/**
	* @brief Set the name of the user who created the trajectory.
	* @param tng_data the trajectory of which to set the user name.
	* @param new_name is a string containing the wanted name.
	* @return TNG_SUCCESS (0) if successful or TNG_CRITICAL (2) if a major
	* error has occured.
	*/
	tng_function_status setFirstUserName(const char *new_name)
	{
		return status = tng_first_user_name_set(traj, new_name);
	}


	/**
	* @brief Get the name of the user who last modified the trajectory.
	* @param tng_data the trajectory of which to get the user name.
	* @param name the string to fill with the name of the user,
	* memory must be allocated before.
	* @param max_len maximum char length of the string, i.e. how much memory has
	* been reserved for name. This includes \0 terminating character.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred (source string longer than destination string).
	*/
	tng_function_status getLastUserName(char *name, const int max_len)
	{
		return status = tng_last_user_name_get(traj,name,max_len);
	}


	/**
	* @brief Set the name of the user who last modified the trajectory.
	* @param new_name is a string containing the wanted name.
	* @return TNG_SUCCESS (0) if successful or TNG_CRITICAL (2) if a major
	* error has occured.
	*/
	tng_function_status setLastUserName(const char *new_name)
	{
		return status = tng_last_user_name_set(traj,new_name);
	}



	/**
	* @brief Get the name of the computer used when creating the trajectory.
	* @param name the string to fill with the name of the computer,
	* memory must be allocated before.
	* @param max_len maximum char length of the string, i.e. how much memory has
	* been reserved for name. This includes \0 terminating character.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred (source string longer than destination string).
	*/
	tng_function_status getFirstComputerName(char *name, const int max_len)
	{
		return status = tng_first_computer_name_get(traj, name, max_len);
	}


	/**
	* @brief Set the name of the computer used when creating the trajectory.
	* @param new_name is a string containing the wanted name.
	* @return TNG_SUCCESS (0) if successful or TNG_CRITICAL (2) if a major
	* error has occured.
	*/
	tng_function_status setFirstComputerName(const char *new_name)
	{
		return status = tng_first_computer_name_set(traj, new_name);
	}


	/**
	* @brief Get the name of the computer used when last modifying the trajectory.
	* @param name the string to fill with the name of the computer,
	* memory must be allocated before.
	* @param max_len maximum char length of the string, i.e. how much memory has
	* been reserved for name. This includes \0 terminating character.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred (source string longer than destination string).
	*/
	tng_function_status getLastComputerName(char *name, const int max_len)
	{
		return status = tng_last_computer_name_get(traj,name,max_len);
	}


	/**
	* @brief Set the name of the computer used when last modifying the trajectory.
	* @param new_name is a string containing the wanted name.
	* @return TNG_SUCCESS (0) if successful or TNG_CRITICAL (2) if a major
	* error has occured.
	*/
	tng_function_status setLastComputerName(const char *new_name)
	{
		return status = tng_last_computer_name_set(traj,new_name);
	}


	/**
	* @brief Get the pgp_signature of the user creating the trajectory.
	* @param signature the string to fill with the signature,
	* memory must be allocated before.
	* @param max_len maximum char length of the string, i.e. how much memory has
	* been reserved for name. This includes \0 terminating character.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred (source string longer than destination string).
	*/
	tng_function_status getFirstSignature(char *signature, const int max_len)
	{
		return status = tng_last_computer_name_get(traj, signature,max_len);
	}


	/**
	* @brief Set the pgp_signature of the user creating the trajectory.
	* @param tng_data the trajectory of which to set the computer name.
	* @param signature is a string containing the pgp_signature.
	* @return TNG_SUCCESS (0) if successful or TNG_CRITICAL (2) if a major
	* error has occured.
	*/
	tng_function_status setFirstSignature(const char *signature)
	{
		return status = tng_first_signature_set(traj, signature);
	}


	/**
	* @brief Get the pgp_signature of the user last modifying the trajectory.
	* @param signature the string to fill with the signature,
	* memory must be allocated before.
	* @param max_len maximum char length of the string, i.e. how much memory has
	* been reserved for name. This includes \0 terminating character.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred (source string longer than destination string).
	*/
	tng_function_status getLastSignature(char *signature, const int max_len)
	{
		return status = tng_first_signature_get(traj, signature, max_len);
	}


	/**
	* @brief Set the pgp_signature of the user last modifying the trajectory.
	* @param signature is a string containing the pgp_signature.
	* @return TNG_SUCCESS (0) if successful or TNG_CRITICAL (2) if a major
	* error has occured.
	*/
	tng_function_status setLastSignature(const char *signature)
	{
		return status = tng_last_signature_set(traj, signature);
	}


	/**
	* @brief Get the name of the forcefield used in the trajectory.
	* @param name the string to fill with the name of the forcefield,
	* memory must be allocated before.
	* @param max_len maximum char length of the string, i.e. how much memory has
	* been reserved for name. This includes \0 terminating character.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred (source string longer than destination string).
	*/
	tng_function_status getForcefieldName(char *name, const int max_len)
	{
		return status = tng_last_signature_get(traj,name,max_len);
	}


	/**
	* @brief Set the name of the forcefield used in the trajectory.
	* @param new_name is a string containing the wanted name.
	* @return TNG_SUCCESS (0) if successful or TNG_CRITICAL (2) if a major
	* error has occured.
	*/
	tng_function_status setForcefieldName(const char *new_name)
	{
		return status = tng_forcefield_name_set(traj, new_name);
	}


	/**
	* @brief Get the medium stride length of the trajectory.
	* @param len is pointing to a value set to the stride length.
	* @return TNG_SUCCESS (0) if successful.
	*/
	tng_function_status getMediumStrideLength(int64_t *len)
	{
		return status = tng_medium_stride_length_get(traj,len);
	}


	/**
	* @brief Set the medium stride length of the trajectory.
	* @param len is the wanted medium stride length.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred.
	*/
	tng_function_status setMediumStrideLength(const int64_t len)
	{
		return status = tng_medium_stride_length_set(traj,len);
	}


	/**
	* @brief Get the long stride length of the trajectory.
	* @param len is pointing to a value set to the stride length.
	* @return TNG_SUCCESS (0) if successful.
	*/
	tng_function_status getLongStrideLength(int64_t *len)
	{
		return status = tng_long_stride_length_get(traj, len);
	}


	/**
	* @brief Set the long stride length of the trajectory.
	* @param len is the wanted long stride length.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred.
	*/
	tng_function_status setLongStrideLength(const int64_t len)
	{
		return status = tng_long_stride_length_set(traj,len);
	}


	/**
	* @brief Get the length of the input file.
	* @param tng_data is the trajectory from which to get the input file length.
	* @param len is pointing to a value set to the file length.
	* @return TNG_SUCCESS (0) if successful.
	*/
	tng_function_status getInputFileLen(int64_t *len)
	{
		return status = tng_input_file_len_get(traj, len);
	}


	/**
	* @brief Get the current number of particles.
	* @param n is pointing to a value set to the number of particles.
	* @details If variable number of particles are used this function will return
	* the number of particles in the current frame set.
	* @return TNG_SUCCESS (0) if successful.
	*/
	tng_function_status getNumParticles(int64_t *n)
	{
		return status = tng_num_particles_get(traj, n);
	}




	/**
	* @brief Get the current total number of molecules.
	* @param n is pointing to a value set to the number of molecules.
	* @details If variable number of particles are used this function will return
	* the total number of molecules in the current frame set.
	* @return TNG_SUCCESS (0) if successful.
	*/
	tng_function_status getNumMolecules(int64_t *n)
	{
		return status = tng_num_molecules_get(traj,n);
	}



	/**
	* @brief Get the number of frames per frame set.
	* per frame set.
	* @param n is pointing to a value set to the number of frames per frame set.
	* @return TNG_SUCCESS (0) if successful.
	*/
	tng_function_status getNumFramesPerFrameSet(int64_t *n)
	{
		return status = tng_num_frames_per_frame_set_get(traj,n);
	}


	/**
	* @brief Get the number of frame sets.
	* @param tng_data is the trajectory from which to get the number of frame sets.
	* @param n is pointing to a value set to the number of frame sets.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred or TNG_CRITICAL (2) if a major error has occured.
	*/
	tng_function_status getNumFrameSets(int64_t *n)
	{
		return status = tng_num_frame_sets_get(traj, n);
	}


	/**
	* @brief Get the current trajectory frame set.
	* @param frame_set_p will be set to point at the memory position of
	* the found frame set.
	* @return TNG_SUCCESS (0) if successful.
	*/
	tng_function_status getCurrentFrameSet(tng_trajectory_frame_set_t *frame_set_p)
	{
		return status = tng_current_frame_set_get(traj, frame_set_p);
	}


	/**
	* @brief Find the requested frame set number.
	* @param tng_data is the trajectory from which to get the frame set.
	* @param nr is the frame set number to search for.
	* @details tng_data->current_trajectory_frame_set will contain the
	* found trajectory if successful.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred or TNG_CRITICAL (2) if a major error has occured.
	*/
	tng_function_status findFrameSetNr(const int64_t nr)
	{
		return status = tng_frame_set_nr_find(traj,nr);
	}


	/**
	* @brief Find the frame set containing a specific frame.
	* @param frame is the frame number to search for.
	* @details tng_data->current_trajectory_frame_set will contain the
	* found trajectory if successful.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred or TNG_CRITICAL (2) if a major error has occured.
	*/
	tng_function_status findFrameSetOfFrame(const int64_t frame)
	{
		return status = tng_frame_set_of_frame_find(traj, frame);
	}


	/**
	* @brief Get the file position of the next frame set in the input file.
	* @param frame_set is the frame set of which to get the position of the
	* following frame set.
	* @param pos is pointing to a value set to the file position.
	* @return TNG_SUCCESS (0) if successful.
	*/
	tng_function_status getFrameSetNextFrameSetFilePos
		(const tng_trajectory_frame_set_t frame_set,int64_t *pos)
	{
		return status = tng_frame_set_next_frame_set_file_pos_get(traj,frame_set,pos );
	}

	/**
	* @brief Get the file position of the previous frame set in the input file.
	* @param frame_set is the frame set of which to get the position of the
	* previous frame set.
	* @param pos is pointing to a value set to the file position.
	* @return TNG_SUCCESS (0) if successful.
	*/
	tng_function_status getFrameSetPrevFrameSetFilePos
		(const tng_trajectory_frame_set_t frame_set,int64_t *pos)
	{
		return status = tng_frame_set_prev_frame_set_file_pos_get(traj, frame_set, pos);
	}


	/**
	* @brief Get the first and last frames of the frame set.
	* @param frame_set is the frame set of which to get the frame range.
	* @param first_frame is set to the first frame of the frame set.
	* @param last_frame is set to the last frame of the frame set.
	* @return TNG_SUCCESS (0) if successful.
	*/
	tng_function_status getFrameSetFrameRange
		(const tng_trajectory_frame_set_t frame_set,
		int64_t *first_frame,
		int64_t *last_frame)
	{
		return status = tng_frame_set_frame_range_get(traj,frame_set, first_frame, last_frame);
	}


	/**
	* @brief Get the molecume name of real particle number (number in mol system).
	* @param nr is the real number of the particle in the molecular system.
	* @param name is a string, which is set to the name of the molecule. Memory
	* must be reserved beforehand.
	* @param max_len is the maximum length of name.
	* @return TNG_SUCCESS (0) if successful or TNG_FAILURE (!) if a minor error
	* has occured.
	*/
	tng_function_status getMoleculeNameOfParticleNr
		(const int64_t nr,char *name,int max_len)
	{
		return status = tng_molecule_name_of_particle_nr_get(traj,nr,name,max_len);

	}


	/**
	* @brief Get the chain name of real particle number (number in mol system).
	* @param nr is the real number of the particle in the molecular system.
	* @param name is a string, which is set to the name of the chain. Memory
	* must be reserved beforehand.
	* @param max_len is the maximum length of name.
	* @return TNG_SUCCESS (0) if successful or TNG_FAILURE (!) if a minor error
	* has occured.
	*/
	tng_function_status getChainNameOfParticleNr
		(const int64_t nr,char *name,int max_len)
	{
		return status = tng_chain_name_of_particle_nr_get(traj, nr, name, max_len);
	}


	/**
	* @brief Get the residue name of real particle number (number in mol system).
	* @param nr is the real number of the particle in the molecular system.
	* @param name is a string, which is set to the name of the residue. Memory
	* must be reserved beforehand.
	* @param max_len is the maximum length of name.
	* @return TNG_SUCCESS (0) if successful or TNG_FAILURE (!) if a minor error
	* has occured.
	*/
	tng_function_status getResidueNameOfParticleNr
		(const int64_t nr,char *name,int max_len)
	{
		return status = tng_residue_name_of_particle_nr_get(traj,nr,name,max_len);
	}


	/**
	* @brief Get the atom name of real particle number (number in mol system).
	* @param nr is the real number of the particle in the molecular system.
	* @param name is a string, which is set to the name of the atom. Memory
	* must be reserved beforehand.
	* @param max_len is the maximum length of name.
	* @return TNG_SUCCESS (0) if successful or TNG_FAILURE (!) if a minor error
	* has occured.
	*/
	tng_function_status getAtomNameOfParticleNr
		(const int64_t nr,char *name,int max_len)
	{
		return status = tng_atom_name_of_particle_nr_get(traj, nr,name,max_len);
	}


	/**
	* @brief Add a particle mapping table.
	* @details Each particle mapping table will be written as a separate block,
	* followed by the data blocks for the corresponding particles. In most cases
	* there is one particle mapping block for each thread writing the trajectory.
	* @details The mapping information is added to the currently active frame set
	* of tng_data
	* @param first_particle_number is the first particle number of this mapping
	* block.
	* @param n_particles is the number of particles in this mapping block.
	* @param mapping_table is a list of the real particle numbers (i.e. the numbers
	* used in the molecular system). The list is n_particles long.
	* @details mapping_table[0] is the real particle number of the first particle
	* in the following data blocks.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred or TNG_CRITICAL (2) if a major error has occured.
	*/
	tng_function_status addParticleMapping
		(const int64_t first_particle_number,
		const int64_t n_particles,
		const int64_t *mapping_table)
	{
		return status = tng_particle_mapping_add(traj,first_particle_number,n_particles,mapping_table );
	}


	/**
	* @brief Read the header blocks from the input_file of tng_data.
	* @details The trajectory blocks must be read separately and iteratively in chunks
	* to fit in memory.
	* @details tng_data->input_file_path specifies
	* which file to read from. If the file (input_file) is not open it will be
	* opened.
	* @param hash_mode is an option to decide whether to use the md5 hash or not.
	* If hash_mode == TNG_USE_HASH the written md5 hash in the file will be
	* compared to the md5 hash of the read contents to ensure valid data.
	* @return TNG_SUCCESS (0) if successful or TNG_CRITICAL (2) if a major
	* error has occured.
	*/
	tng_function_status readFileHeaders(const tng_hash_mode hash_mode)
	{
		return status = tng_file_headers_read(traj, hash_mode);
	}


	/**
	* @brief Write the header blocks to the output_file of tng_data.
	* @details The trajectory blocks must be written separately and iteratively in chunks
	* to fit in memory.
	* @details tng_data->output_file_path
	* specifies which file to write to. If the file (output_file) is not open it
	* will be opened.
	* @param hash_mode is an option to decide whether to use the md5 hash or not.
	* If hash_mode == TNG_USE_HASH an md5 hash for each header block will be generated.
	* @return TNG_SUCCESS (0) if successful or TNG_CRITICAL (2) if a major
	* error has occured.
	*/
	tng_function_status writeFileHeaders(const tng_hash_mode hash_mode)
	{
		return status = tng_file_headers_write(traj, hash_mode);
	}



	/**
	* @brief Read one (the next) block (of any kind) from the input_file of tng_data.
	* which file to read from. If the file (input_file) is not open it will be
	* opened.
	* @param block_data is a pointer to the struct which will be populated with the
	* data.
	* @details If block_data->input_file_pos > 0 it is the position from where the
	* reading starts otherwise it starts from the current position.
	* @param hash_mode is an option to decide whether to use the md5 hash or not.
	* If hash_mode == TNG_USE_HASH the written md5 hash in the file will be
	* compared to the md5 hash of the read contents to ensure valid data.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred or TNG_CRITICAL (2) if a major error has occured.
	*/
	tng_function_status readNextBlock(const tng_hash_mode hash_mode, tng_gen_block_t block_data)
	{
		return status = tng_block_read_next(traj,block_data, hash_mode);
	}



	/**
	* @brief Read one (the next) frame set, including mapping and related data blocks
	* from the input_file of tng_data.
	* which file to read from. If the file (input_file) is not open it will be
	* opened.
	* @param hash_mode is an option to decide whether to use the md5 hash or not.
	* If hash_mode == TNG_USE_HASH the written md5 hash in the file will be
	* compared to the md5 hash of the read contents to ensure valid data.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred or TNG_CRITICAL (2) if a major error has occured.
	*/
	tng_function_status readNextFrameSet(const tng_hash_mode hash_mode)
	{
		return status = tng_frame_set_read_next(traj, hash_mode);
	}


	/**
	* @brief Write one frame set, including mapping and related data blocks
	* to the output_file of tng_data.
	* @details  tng_data->output_file_path specifies
	* which file to write to. If the file (output_file) is not open it will be
	* opened.
	* @param hash_mode is an option to decide whether to use the md5 hash or not.
	* If hash_mode == TNG_USE_HASH an md5 hash for each header block will be generated.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred or TNG_CRITICAL (2) if a major error has occured.
	*/
	tng_function_status writeFrameSet(const tng_hash_mode hash_mode)
	{
		return status = tng_frame_set_write(traj, hash_mode);
	}


	/**
	* @brief Create and initialise a frame set.
	* @param first_frame is the first frame of the frame set.
	* @param n_frames is the number of frames in the frame set.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred or TNG_CRITICAL (2) if a major error has occured.
	*/
	tng_function_status newFrameSet(const int64_t first_frame,
		const int64_t n_frames)
	{
		return status =  tng_frame_set_new(traj, first_frame,n_frames);
	}


	/**
	* @brief Add a non-particle dependent data block.
	* @param id is the block ID of the block to add.
	* @param block_name is a descriptive name of the block to add
	* @param datatype is the datatype of the data in the block (e.g. int/float)
	* @param block_type_flag indicates if this is a non-trajectory block (added
	* directly to tng_data) or if it is a trajectory block (added to the
	* frame set)
	* @param n_frames is the number of frames of the data block (automatically
	* set to 1 if adding a non-trajectory data block)
	* @param n_values_per_frame is how many values a stored each frame (e.g. 9
	* for a box shape block)
	* @param stride_length is how many frames are between each entry in the
	* data block
	* @param codec_id is the ID of the codec to compress the data.
	* @param new_data is an array of data values to add.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred or TNG_CRITICAL (2) if a major error has occured.
	*/
	tng_function_status addDataBlock(const int64_t id,
		const char *block_name,
		const tng_data_type datatype,
		const tng_block_type block_type_flag,
		int64_t n_frames,
		const int64_t n_values_per_frame,
		const int64_t stride_length,
		const int64_t codec_id,
		void *new_data)
	{
		return status = tng_data_block_add(traj, id,block_name,
			datatype,block_type_flag, n_frames,
			n_values_per_frame, stride_length,
			codec_id, new_data);
	}


	/**
	* @brief Add a particle dependent data block.
	* @param id is the block ID of the block to add.
	* @param block_name is a descriptive name of the block to add
	* @param datatype is the datatype of the data in the block (e.g. int/float)
	* @param block_type_flag indicates if this is a non-trajectory block (added
	* directly to tng_data) or if it is a trajectory block (added to the
	* frame set)
	* @param n_frames is the number of frames of the data block (automatically
	* set to 1 if adding a non-trajectory data block)
	* @param n_values_per_frame is how many values a stored each frame (e.g. 9
	* for a box shape block)
	* @param stride_length is how many frames are between each entry in the
	* data block
	* @param first_particle_number is the number of the first particle stored
	* in this data block
	* @param n_particles is the number of particles stored in this data block
	* @param codec_id is the ID of the codec to compress the data.
	* @param new_data is an array of data values to add.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred or TNG_CRITICAL (2) if a major error has occured.
	*/
	tng_function_status addParticleDataBlock(const int64_t id,
		const char *block_name,
		const tng_data_type datatype,
		const tng_block_type block_type_flag,
		int64_t n_frames,
		const int64_t n_values_per_frame,
		const int64_t stride_length,
		const int64_t first_particle_number,
		const int64_t n_particles,
		const int64_t codec_id,
		void *new_data)
	{
		return status = tng_particle_data_block_add(traj,id,	block_name,
			datatype, block_type_flag, n_frames,n_values_per_frame,
			stride_length,first_particle_number,n_particles,
			codec_id, new_data);
	}


	/**
	* @brief Write data of one trajectory frame to the output_file of tng_data.
	* @param frame_nr is the index number of the frame to write.
	* @param block_id is the ID of the data block to write the data to.
	* @param data is an array of data to write. The length of the array should
	* equal n_values_per_frame.
	* @param hash_mode is an option to decide whether to use the md5 hash or not.
	* If hash_mode == TNG_USE_HASH the written md5 hash in the file will be
	* compared to the md5 hash of the read contents to ensure valid data.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred or TNG_CRITICAL (2) if a major error has occured.
	*/
	tng_function_status writeFrameData(const int64_t frame_nr,
		const int64_t block_id,
		const void *data,
		const tng_hash_mode hash_mode)
	{
		return status = tng_frame_data_write(traj,frame_nr,block_id,data,hash_mode);
	}


	/**
	* @brief Write particle data of one trajectory frame to the output_file of
	* tng_data.
	* @param frame_nr is the index number of the frame to write.
	* @param block_id is the ID of the data block to write the data to.
	* @param val_first_particle is the number of the first particle in the data
	* array.
	* @param val_n_particles is the number of particles in the data array.
	* @param data is a 1D-array of data to write. The length of the array should
	* equal n_particles * n_values_per_frame.
	* @param hash_mode is an option to decide whether to use the md5 hash or not.
	* If hash_mode == TNG_USE_HASH the written md5 hash in the file will be
	* compared to the md5 hash of the read contents to ensure valid data.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred or TNG_CRITICAL (2) if a major error has occured.
	*/
	tng_function_status writeFrameParticleData(const int64_t frame_nr,
		const int64_t block_id,
		const int64_t val_first_particle,
		const int64_t val_n_particles,
		const void *data,
		const tng_hash_mode hash_mode)
	{
		return status = tng_frame_particle_data_write(traj,frame_nr,block_id,val_first_particle,val_n_particles,data,hash_mode);
	}


	/**
	* @brief Free data is an array of values (2D).
	* @param values is the 2D array to free and will be set to 0 afterwards.
	* @param n_frames is the number of frames in the data array.
	* @param n_values_per_frame is the number of values per frame in the data array.
	* @param type is the data type of the data in the array (e.g. int/float/char).
	* @return TNG_SUCCESS (0) if successful.
	*/
	tng_function_status freeDataValues(union data_values **values,
		const int64_t n_frames,
		const int64_t n_values_per_frame,
		const tng_data_type type)
	{
		return status = tng_data_values_free(traj, values, n_frames,n_values_per_frame,type);
	}


	/**
	* @brief Free data is an array of values (3D).
	* @param values is the array to free and will be set to 0 afterwards.
	* @param n_frames is the number of frames in the data array.
	* @param n_particles is the number of particles in the data array.
	* @param n_values_per_frame is the number of values per frame in the data array.
	* @param type is the data type of the data in the array (e.g. int/float/char).
	* @return TNG_SUCCESS (0) if successful.
	*/
	tng_function_status freeParticleDataValues(union data_values ***values,
		const int64_t n_frames,
		const int64_t n_particles,
		const int64_t n_values_per_frame,
		const tng_data_type type)
	{
		return status = tng_particle_data_values_free(traj, values,n_frames,n_particles,n_values_per_frame,type);
	}


	/**
	* @brief Retrieve non-particle data, from the last read frame set.
	* which file to read from. If the file (input_file) is not open it will be
	* opened.
	* @param block_id is the id number of the particle data block to read.
	* @param values is a pointer to a 2-dimensional array (memory unallocated), which
	* will be filled with data. The array will be sized
	* (n_frames * n_values_per_frame).
	* Since ***values is allocated in this function it is the callers
	* responsibility to free the memory.
	* @param n_frames is set to the number of particles in the returned data. This is
	* needed to properly reach and/or free the data afterwards.
	* @param n_values_per_frame is set to the number of values per frame in the data.
	* This is needed to properly reach and/or free the data afterwards.
	* @param type is set to the data type of the data in the array.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred or TNG_CRITICAL (2) if a major error has occured.
	*/
	tng_function_status getData(const int64_t block_id,
	union data_values ***values,
		int64_t *n_frames,
		int64_t *n_values_per_frame,
		tng_data_type *type)
	{
		return status = tng_data_get(traj,block_id,values,n_frames,n_values_per_frame,type);
	}


	/**
	* @brief Read and retrieve non-particle data, in a specific interval.
	* which file to read from. If the file (input_file) is not open it will be
	* opened.
	* @param block_id is the id number of the particle data block to read.
	* @param start_frame_nr is the index number of the first frame to read.
	* @param end_frame_nr is the index number of the last frame to read.
	* @param hash_mode is an option to decide whether to use the md5 hash or not.
	* If hash_mode == TNG_USE_HASH the md5 hash in the file will be
	* compared to the md5 hash of the read contents to ensure valid data.
	* @param values is a pointer to a 2-dimensional array (memory unallocated), which
	* will be filled with data. The array will be sized
	* (n_frames * n_values_per_frame).
	* Since ***values is allocated in this function it is the callers
	* responsibility to free the memory.
	* @param n_values_per_frame is set to the number of values per frame in the data.
	* This is needed to properly reach and/or free the data afterwards.
	* @param type is set to the data type of the data in the array.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred or TNG_CRITICAL (2) if a major error has occured.
	*/
	tng_function_status getDataInterval(const int64_t block_id,
		const int64_t start_frame_nr,
		const int64_t end_frame_nr,
		const tng_hash_mode hash_mode,
	union data_values ***values,
		int64_t *n_values_per_frame,
		tng_data_type *type)
	{
		return status = tng_data_interval_get(traj, block_id,start_frame_nr,end_frame_nr,hash_mode,values,n_values_per_frame,type);
	}


	/**
	* @brief Retrieve particle data, from the last read frame set.
	* @details The particle dimension of the returned values array is translated
	* to real particle numbering, i.e. the numbering of the actual molecular
	* system.
	* specifies which file to read from. If the file (input_file) is not open it
	* will be opened.
	* @param block_id is the id number of the particle data block to read.
	* @param values is a pointer to a 3-dimensional array (memory unallocated), which
	* will be filled with data. The array will be sized
	* (n_frames * n_particles * n_values_per_frame).
	* Since ****values is allocated in this function it is the callers
	* responsibility to free the memory.
	* @param n_frames is set to the number of particles in the returned data. This is
	* needed to properly reach and/or free the data afterwards.
	* @param n_particles is set to the number of particles in the returned data. This is
	* needed to properly reach and/or free the data afterwards.
	* @param n_values_per_frame is set to the number of values per frame in the data.
	* This is needed to properly reach and/or free the data afterwards.
	* @param type is set to the data type of the data in the array.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred or TNG_CRITICAL (2) if a major error has occured.
	*/
	tng_function_status getParticleData(const int64_t block_id,
	union data_values ****values,
		int64_t *n_frames,
		int64_t *n_particles,
		int64_t *n_values_per_frame,
		tng_data_type *type)
	{
		return status = (tng_particle_data_get(traj, block_id, values, n_frames,
			n_particles, n_values_per_frame, type));
	}



	/**
	* @brief Read and retrieve particle data, in a specific interval.
	* @details The particle dimension of the returned values array is translated
	* to real particle numbering, i.e. the numbering of the actual molecular
	* system.
	* @param block_id is the id number of the particle data block to read.
	* @param start_frame_nr is the index number of the first frame to read.
	* @param end_frame_nr is the index number of the last frame to read.
	* @param hash_mode is an option to decide whether to use the md5 hash or not.
	* If hash_mode == TNG_USE_HASH the md5 hash in the file will be
	* compared to the md5 hash of the read contents to ensure valid data.
	* @param values is a pointer to a 3-dimensional array (memory unallocated), which
	* will be filled with data. The array will be sized
	* (n_frames * n_particles * n_values_per_frame).
	* Since ****values is allocated in this function it is the callers
	* responsibility to free the memory.
	* @param n_particles is set to the number of particles in the returned data. This is
	* needed to properly reach and/or free the data afterwards.
	* @param n_values_per_frame is set to the number of values per frame in the data.
	* This is needed to properly reach and/or free the data afterwards.
	* @param type is set to the data type of the data in the array.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred or TNG_CRITICAL (2) if a major error has occured.
	*/
	tng_function_status getParticleDataInterval(const int64_t block_id,
		const int64_t start_frame_nr,
		const int64_t end_frame_nr,
		const tng_hash_mode hash_mode,
	union data_values ****values,
		int64_t *n_particles,
		int64_t *n_values_per_frame,
		tng_data_type *type)
	{
		return status = (tng_particle_data_interval_get(traj, block_id, start_frame_nr,
			end_frame_nr, hash_mode, values,
			n_particles, n_values_per_frame,
			type));
	}


	/** @brief Get the date and time of initial file creation in ISO format (string).
	*  @param time is a pointer to the string in which the date will be stored. Memory
	must be reserved beforehand.
	* @return TNG_SUCCESS (0) if successful.
	*/
	tng_function_status getStrTime(char *time)
	{
		return status = tng_time_get_str(traj, time);
	}


};





class Molecule
{
private:

	tng_molecule_t mol;
	Trajectory * traj;
	tng_function_status status;
public:
	friend class Trajectory;
	//Constructor
	Molecule(Trajectory * trajectory)
	{
		traj = trajectory;
		
		//status = tng_molecule_init(traj->traj,mol);
	}
	/**
     *@Dose nothing, use ~TngMolecule()
	*/
	~Molecule()
	{
		status = tng_molecule_destroy(traj->traj,mol);
	}
		//! Status
	tng_function_status getStatus()
	{ return status; }


	/**
	* @brief Set the name of a molecule.
	* @param new_name is a string containing the wanted name.
	* @return TNG_SUCCESS (0) if successful or TNG_CRITICAL (2) if a major
	* error has occured.
	*/
	tng_function_status setMoleculeName(const char *new_name)
	{
		return status = tng_molecule_name_set(traj->traj,mol,new_name);
	}

	/**
	* @brief Get the count of a molecule.
	* @param cnt is a pointer to the variable to be populated with the count.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred or TNG_CRITICAL (2) if a major error has occured.
	*/
	tng_function_status getMoleculeCnt(int64_t *cnt)
	{
		return status = tng_molecule_cnt_get(traj->traj,mol,cnt);
	}

	/**
	* @brief Set the count of a molecule.
	* @param cnt is the number of instances of this molecule.
	* @return TNG_SUCCESS (0) if successful, TNG_FAILURE (1) if a minor error
	* has occurred or TNG_CRITICAL (2) if a major error has occured.
	*/
	tng_function_status setMoleculeCnt(int64_t cnt)
	{
		return status = tng_molecule_cnt_set(traj->traj,mol,cnt);
	}

	/**
	* @brief Add a chain to a molecule.
	* @param name is a string containing the name of the chain.
	* @param chain s a pointer to the newly created chain.
	* @return TNG_SUCCESS (0) if successful or TNG_CRITICAL (2) if a major
	* error has occured.
	*/
	tng_function_status addMoleculeChain(const char *name,Chain *chain)
	{
		return status = tng_molecule_chain_add(traj->traj,mol,name,&chain->chain);
	}

};

	tng_function_status Trajectory::addMolecule(const char *name, Molecule_t molecule)
{
	return status = tng_molecule_add(traj,name, &molecule->mol);
}

class Atom
{
private:
	tng_atom_t atom;
	Trajectory * traj;
	tng_function_status status;
public:
	friend class Residue;
	//constructor
	Atom(Trajectory * trajectory)
	{
		traj = trajectory;
	}
	//deonstructor
	/**
     *@Dose nothing, use ~TngMolecule()
	*/
		~Atom()
	{
		//delete atom;
	}
			//! Status
	tng_function_status getStatus()
	{ return status; }
	/**
	* @brief Set the name of an atom.
	* @param new_name is a string containing the wanted name.
	* @return TNG_SUCCESS (0) if successful or TNG_CRITICAL (2) if a major
	* error has occured.
	*/
	tng_function_status setAtomName(const char *new_name)
	{
		return status = tng_atom_name_set(traj->traj, atom , new_name);
	}

	/**
	* @param tng_data is the trajectory data container containing the atom.
	* @param atom is the atom to change.
	* @param new_type is a string containing the atom type.
	* @return TNG_SUCCESS (0) if successful or TNG_CRITICAL (2) if a major
	* error has occured.
	*/
	tng_function_status setAtomType(const char *new_type)
	{
		return status = tng_atom_type_set(traj->traj, atom, new_type);
	}
	};

class Residue
{
	private:
	tng_residue_t residue;
	Trajectory * traj;
	tng_function_status status;
public:
	friend class Chain;
	//constructor
	Residue(Trajectory  * trajectory)
	{
		traj = trajectory;
	}
	//deonstructor
	/**
     *@Dose nothing, use ~TngMolecule()
	*/
		~Residue()
	{
		//delete residue;
	}
	//! Status
	tng_function_status getStatus()
	{ return status; }
	/**
	* @brief Set the name of a residue.
	* @param residue is the residue to rename.
	* @param new_name is a string containing the wanted name.
	* @return TNG_SUCCESS (0) if successful or TNG_CRITICAL (2) if a major
	* error has occured.
	*/
	tng_function_status setResidueName(const char *new_name)
	{
		return status = tng_residue_name_set(traj->traj, residue,new_name);
	}


	/**
	* @brief Add an atom to a residue.
	* @param atom_name is a string containing the name of the atom.
	* @param atom_type is a string containing the atom type of the atom.
	* @param atom is a pointer to the newly created atom.
	* @return TNG_SUCCESS (0) if successful or TNG_CRITICAL (2) if a major
	* error has occured.
	*/
	tng_function_status addResidueAtom(const char *atom_name,
		const char *atom_type,
		Atom * atom)
	{
		return status = tng_residue_atom_add(traj->traj,residue,atom_name,atom_type,&atom->atom);
	}
	};

class Chain
{
	private:
	tng_chain_t chain;
	Trajectory * traj;
	tng_function_status status;
public:
	friend class Molecule;
	//constructor
	Chain(Trajectory * trajectory)
	{
		traj = trajectory;
	}
	//deonstructor
	/**
     *@Dose nothing, use ~TngMolecule()
	*/
		~Chain()
	{
		//delete chain;
	}
	//! Status
	tng_function_status getStatus()
	{ return status; }
	/**
	* @brief Set the name of a chain.
	* @param new_name is a string containing the wanted name.
	* @return TNG_SUCCESS (0) if successful or TNG_CRITICAL (2) if a major
	* error has occured.
	*/
	tng_function_status setChainName(const char *new_name)
	{
		return status  = tng_chain_name_set(traj->traj, chain, new_name);
	}


	/**
	* @brief Add a residue to a chain.
	* @param name is a string containing the name of the residue.
	* @param residue is a pointer to the newly created residue.
	* @return TNG_SUCCESS (0) if successful or TNG_CRITICAL (2) if a major
	* error has occured.
	*/
	tng_function_status addChainResidue(const char *name,
		Residue * residue)
	{
		return status = tng_chain_residue_add(traj->traj,chain,name,&residue->residue);
	}
};

}
#endif 