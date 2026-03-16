#pragma once

#include <cstdint>

// Paul Bourke's marching cubes lookup tables.
// edgeTable[i]: 12-bit mask of which edges are intersected for cube config i.
// triTable[i * 16 + j]: edge indices forming triangles, terminated by -1.

extern const uint32_t MC_EDGE_TABLE[256];
extern const int32_t MC_TRI_TABLE[256 * 16];