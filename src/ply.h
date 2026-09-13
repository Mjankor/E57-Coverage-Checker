// Getting the answer back out: the voxels, or the shell, as a file something
// else can open.
//
// PLY, binary little-endian, because the deliverable has to land in whatever the
// surveyor already uses and PLY is the one point format every one of them reads
// without a plugin or a licence — CloudCompare, MeshLab, Recap, Cyclone, Blender.
// LAS would be the survey-native choice and is the obvious next one to add; it
// needs a header, a projection and a scale/offset model, which is a writer rather
// than the hundred lines below.
//
// COORDINATES ARE DOUBLES, and that is the whole reason this is not three lines
// of fprintf. A carve's points live as float offsets from a double origin (see
// lod::StorePoint) precisely because a georeferenced site sits at UTM magnitudes
// where float32 resolves about a metre — so writing float world coordinates would
// throw away the precision the rest of the pipeline is built to keep. PLY has a
// `double` property type and the readers above honour it, so the file carries
// absolute position at full precision and needs no shift agreed out of band.

#pragma once

#include "lod.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ply {

// Lines written into the file's header as `comment` records, so a saved cloud
// says what produced it. PLY comments survive a round trip through every reader
// tested, and a file that cannot say which run and which settings made it is not
// evidence of anything.
struct Note {
    std::vector<std::string> comments;
};

// Writes `pts` as a point cloud. Positions are `origin` plus each point's own
// float offset, in metres; colours come through as they are.
//
// Returns false and sets `err` on any failure, having written nothing it can
// help — a half-written PLY that looks openable is worse than none.
bool writePoints(const std::string& path,
                 const std::vector<lod::StorePoint>& pts,
                 const double origin[3],
                 const Note& note,
                 std::string& err);

} // namespace ply
