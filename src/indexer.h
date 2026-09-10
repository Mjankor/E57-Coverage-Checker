// Building a point store from an E57 corpus.
//
// Two entry points, matching the two things the app has to do with a thousand
// files:
//
//   survey()  Headers only — pose, prototype, declared bounds, metadata
//             classification. No point decoding, so a thousand files take
//             seconds and the setup layout can be drawn immediately while the
//             real work happens behind it.
//
//   build()   The bounded-memory octree build. Memory stays flat regardless of
//             corpus size, which is the whole point: the store this produces is
//             routinely larger than RAM.
//
// How build() stays bounded:
//
//   1. A top tree is built in memory over levels 0..chunkLevel-1, with
//      `spillAtMaxLevel` set so each node keeps only its occupancy-grid sample
//      and everything else spills.
//   2. Spilled points are appended to one file per occupied cell at
//      chunkLevel, through small write buffers. This is the bulk of the data
//      and it never accumulates in memory.
//   3. Each chunk is then built into a subtree on its own and appended to the
//      store, linked under the top tree node it belongs to. Only one chunk is
//      resident at a time.
//
// chunkLevel is chosen so a chunk is roughly `targetPointsPerChunk`, so peak
// memory is bounded by that rather than by the corpus.

#pragma once

#include "e57.h"
#include "frame.h"
#include "lod.h"
#include "point_store.h"
#include "scan_check.h"

#include <map>
#include <functional>
#include <string>
#include <vector>

namespace indexer {

// Return false from a progress callback to cancel. Cancellation is checked
// between scans and between chunks, so it is prompt without leaving a
// half-written store presented as complete.
using ProgressFn = std::function<bool(const std::string& stage, uint64_t done,
                                      uint64_t total)>;

struct ScanRef {
    std::string path;          // file it came from
    size_t      scanIndex = 0; // index within that file
    std::string name;
    std::string guid;

    e57::Pose pose;
    bool      hasPose = false;
    uint64_t  recordCount = 0;

    check::Kind kind = check::Kind::Ambiguous;
    std::string status;
    // Whether this scan can be drawn at all, which is the only question the point
    // store has to ask: it needs positions, and nothing else.
    //
    // False ONLY when the prototype carries no position fields — neither
    // cartesian nor spherical — because then there is nothing to put in the
    // store. Readable from the header, so it costs nothing.
    //
    // It is deliberately NOT a verdict on whether the cloud came from one setup.
    // It used to be: a scan the range-spread heuristic called merged was excluded,
    // and that threw away real data — every scan of the job this was built against
    // measured over the threshold, so the store came out empty and the viewer drew
    // nothing. Drawing points does not require knowing where they were seen from.
    // The stage that does require it, the visibility carve, never reads this and
    // has its own guard (rimg::Options::minRoundTripFraction). See the note at the
    // top of scan_check.h for what the heuristic actually measures.
    //
    // check::Kind still travels alongside, so the UI can label a scan without
    // anything dropping it.
    bool        usable = false;

    // The range-spread heuristic came out over its threshold. Reported — it
    // becomes store::kScanLooksMerged — and acted on by nothing.
    bool        looksMerged = false;

    // Whether the scan's pose must be applied to its points, as the survey
    // decided it — viewer::FrameDecision::applyPose().
    //
    // Carried here so build() does not decide it again. Deciding it needs a sample
    // of the points, so every caller that re-derives it pays a full decode pass;
    // build called streamScan twice and streamScan decided it each time, which was
    // two passes per scan to re-learn something the survey had already worked out
    // from a sample it had in hand.
    //
    // A header-only survey sets this from the pose alone (identity or not), which
    // is what it can know without reading points. That is the same answer
    // decideFrame gives for an identity pose and its documented default otherwise.
    bool        frameApplyPose = false;

    // The setup position in the file's coordinate system. Available from
    // headers alone, which is what makes the fast open possible.
    double setup[3] = {0, 0, 0};

    // Declared extent, transformed into the file frame. Absent for scans whose
    // files omit cartesianBounds, in which case build() samples for bounds.
    bool   hasExtent = false;
    double lo[3] = {0, 0, 0};
    double hi[3] = {0, 0, 0};
};

struct Survey {
    std::vector<ScanRef>     scans;
    std::vector<std::string> errors;      // one per unreadable file
    size_t                   filesRead = 0;
    // Of those, how many were answered from the cache rather than checked. Worth
    // reporting: on an incremental corpus it is nearly all of them, and a figure
    // that suddenly drops to zero is the visible symptom of a cache that is
    // being invalidated by something — a re-written file, a moved directory.
    size_t                   filesFromCache = 0;

    // Union of declared extents and setup positions. `extentComplete` is false
    // when any usable scan lacked declared bounds.
    bool   hasBounds = false;
    bool   extentComplete = true;
    double lo[3] = {0, 0, 0};
    double hi[3] = {0, 0, 0};

    size_t usableCount() const;
    uint64_t totalPoints() const;
};

struct SurveyOptions {
    // Run the geometric merged-cloud test. Decodes a sample per scan, so it is
    // far slower than a header read — off for the fast open, on for the build.
    bool classify = false;

    // Files checked at once. 0 takes the hardware's count.
    //
    // Checking a file shares nothing with checking another — its own reader, its
    // own sample, its own verdict — so this scales with cores until the disk runs
    // out of bandwidth. Measured on eight copies of a 154 MB scan, warm: 1.49 s
    // on one thread, 0.43 s on four. It is capped at the core count rather than
    // obeyed blindly, because eight threads on four cores measured slower than
    // four.
    //
    // The answer does not depend on it. Per-file results land in a slot per file
    // and are merged in path order afterwards, so the scan list, the bounds and
    // the error list come out identical at any thread count — there is a test.
    unsigned threads = 0;

    // Where to remember per-file verdicts between runs. Empty disables it.
    //
    // Checking a file depends on NOTHING but that file, so the result keeps as
    // long as the file does not change. That matters because the octree store is
    // cached under a key covering the whole corpus — add one scan to a thousand
    // and the key moves, so the store is rebuilt and every scan re-checked. The
    // rebuild is unavoidable, an octree over the corpus being a corpus-wide
    // thing; re-checking the 999 unchanged files is not.
    //
    // Only consulted and only written when `classify` is set, because that is
    // the expensive path and the only one worth remembering: a header-only
    // survey costs nothing and reaches a different (cheaper) verdict, which must
    // never be confused with this one.
    //
    // A file is identified by its path, size and modification time, so an edited
    // or replaced scan is re-checked rather than believed. A cache that cannot
    // be read, or is of the wrong version, or has an entry that does not parse,
    // is treated as absent — the only cost of that is doing the work.
    std::string cachePath;
};

// Per-file verdicts remembered between runs. See SurveyOptions::cachePath.
//
// Exposed for testing and for a caller that wants to manage the file itself;
// survey() loads and saves it on its own when given a path.
struct SurveyCache {
    // Missing file is not an error — it is an empty cache. False means the file
    // existed and could not be used, which is also not fatal to a survey.
    bool load(const std::string& path);
    bool save(const std::string& path) const;

    // Entries held. What a survey did with the cache is reported through
    // Survey::filesFromCache rather than from here, because survey() loads its
    // own copy and the caller never sees it.
    size_t entries() const;

    struct Entry {
        uint64_t             size = 0;
        int64_t              mtime = 0;
        std::vector<ScanRef> scans;
        bool                 extentComplete = true;
    };
    // Keyed by path. Public so a test can reach in; the invariant that matters
    // is checked on use rather than on insertion.
    std::map<std::string, Entry> byPath;
};

Survey survey(const std::vector<std::string>& paths, const SurveyOptions& opt,
              const ProgressFn& progress);

// ---------------------------------------------------------------------------

struct BuildOptions {
    lod::BuildOptions tree;
    // Where spill files go. Defaults to the store's directory.
    std::string chunkDir;
    // Peak memory during the per-chunk phase is roughly this many points.
    // 20 M points is ~400 MB of StorePoint.
    uint64_t targetPointsPerChunk = 20000000;
    // 0 picks a level from the corpus size.
    uint8_t  chunkLevel = 0;

    // Chunks built at once. 0 derives it — see maxResidentPoints.
    //
    // Building a chunk is independent work: its own spill file, its own subtree,
    // sharing nothing. It is also about 70 per cent of a build, so this is the
    // stage worth spreading. Appending a finished subtree to the store is NOT
    // independent — node indices depend on append order — so chunks are built in
    // parallel and appended serially in cell order, which is the same order the
    // serial build used. The store comes out byte-identical whatever this is set
    // to; there is a test.
    unsigned chunkThreads = 0;

    // Ceiling on points held in memory at once while chunks are built, which is
    // what bounds this stage's footprint.
    //
    // The whole build exists to keep memory flat regardless of corpus size, and
    // building several chunks at once spends some of that: peak is roughly
    // chunkThreads x targetPointsPerChunk points, at 20 bytes each. So the thread
    // count is derived from this budget rather than from the core count alone, and
    // a larger targetPointsPerChunk buys fewer threads rather than more memory.
    //
    // 80 M points is about 1.6 GB of StorePoint — four 20 M-point chunks, which on
    // a machine with any real amount of RAM is nothing, and on a small one the
    // derivation drops to one chunk and the old behaviour.
    uint64_t maxResidentPoints = 80000000;
    // 0 means every point. Set it to cap a store's size on disk.
    uint64_t maxPointsPerScan = 0;
};

struct BuildStats {
    uint64_t pointsRead = 0;
    uint64_t pointsStored = 0;
    uint64_t spilled = 0;
    // Points discarded because a node was full at the depth limit. Should be
    // zero at any sane depth; reported rather than hidden, because silently
    // losing points would look like missing coverage.
    uint64_t dropped = 0;
    // Points that fell outside the root bounds and are therefore absent from
    // the store. Should be zero; reported rather than fatal, because failing a
    // whole corpus over a stray point is worse than saying how many were lost.
    uint64_t outsideRoot = 0;
    uint64_t nodes = 0;
    size_t   chunks = 0;
    uint8_t  chunkLevel = 0;
    uint64_t storeBytes = 0;
};

bool build(const Survey& s, const std::string& storePath, const BuildOptions& opt,
           BuildStats& stats, const ProgressFn& progress, std::string& err);

// Chooses the chunk level so a chunk holds about `targetPointsPerChunk`.
uint8_t chooseChunkLevel(uint64_t totalPoints, uint64_t targetPointsPerChunk);

// The cell index at `level` containing a point, and optionally that cell's
// bounds. The index packs three bits per level, most significant first.
uint32_t cellIndexOf(const lod::Aabb& root, uint8_t level,
                     float x, float y, float z, lod::Aabb* outBounds = nullptr);

} // namespace indexer
