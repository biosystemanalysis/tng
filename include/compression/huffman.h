/* This code is part of the tng compression routines.
 *
 * Written by Daniel Spangberg
 * Copyright (c) 2010, 2013, The GROMACS development team.
 *
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 */


#ifndef HUFFMAN_H
#define HUFFMAN_H

void Ptngc_comp_conv_to_huffman(unsigned int *vals, int nvals,
			  unsigned int *dict, int ndict,
			  unsigned int *prob,
			  unsigned char *huffman,
			  int *huffman_len,
			  unsigned char *huffman_dict,
			  int *huffman_dictlen,
			  unsigned int *huffman_dict_unpacked,
			  int *huffman_dict_unpackedlen);

void Ptngc_comp_conv_from_huffman(unsigned char *huffman,
			    unsigned int *vals, int nvals,
			    int ndict,
			    unsigned char *huffman_dict,
			    int huffman_dictlen,
			    unsigned int *huffman_dict_unpacked,
			    int huffman_dict_unpackedlen);

#endif