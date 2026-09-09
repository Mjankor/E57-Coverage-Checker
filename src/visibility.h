// The visibility filter as a job: files in, a drawable answer out.
//
// `carve.h` is the method and `range_image.h` is its input; this is the thing
// that runs them over a corpus and hands back something a window can show. It
// exists so the CLI and the app drive exactly the same code — the app has no
// business owning the pipeline, and a result you can only see in a terminal is
// not much of an answer to "which space did the scanners miss?".
//
// Two problems stand between a finished carve and a picture, and both are
// solved here rather than in the renderer:
//
//   Volume. A site 40 m across at 5 cm is order 10^8 voxels in range, and most
//   of them are unknown simply because they are outside the building. Drawing
//   that is neither possible nor useful. What is worth looking at is the
//   frontier: unknown voxels that touch space some setup could see. That is
//   where coverage stops — the mouth of a shadow, the far edge of the range
//   spheres — and since an opaque blob hides its own interior anyway, the
//   frontier looks the same as the solid volume from outside it while costing
//   area instead of volume. `solid` turns the reduction off.
//
//   Count. Even a frontier can exceed what is sensible to upload, so there is
//   a cap. It is applied by hashing each voxel's position on the global lattice
//   and keeping those below a threshold that halves whenever the cap is hit.
//   That makes the sample deterministic and independent of tile size and of the
//   order tiles were carved in — the same run twice, or at a different
//   `tileVoxels`, selects the same voxels.
//
// What this does NOT yet do is classify. Everything it returns is unknown space
// in range of a setup; separating occlusion shadows inside the site from the
// open air outside it needs the connectivity pass in DESIGN.md, and until that
// exists the result has to be read as "not observed", not as "void".

#pragma once

#include "carve.h"
#include "lod.h"
#include "range_image.h"
#include "voids.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace vis {

// Which region the question covers. See carve::Domain for why this matters more
// than any other single setting.
enum class DomainMode {
    // Everything within maxRange of any setup. Honest, and mostly outdoors.
    RangeSpheres,
    // A box around what the scans actually returned, grown by domainMargin.
    // The default, because a scan of a building interior otherwise spends
    // almost all of its answer on the sky and the neighbours' gardens.
    //
    // A box is the first approximation of the right shape. The right shape is a
    // shrinkwrap of the returns, which for an interior job is the building and
    // which would also drop the corners of this box that no scan ever reached.
    // carve::Domain is the seam that will be cut along.
    MeasuredExtent,
};

struct Options {
    double   voxelSize  = 0.05;
    double   maxRange   = 45.0;
    uint32_t tileVoxels = 128;
    // Stop after this many tiles. 0 runs the whole domain; anything else is a
    // sample, and `Result::partial` says so.
    uint64_t maxTiles   = 0;

    // Keep only unknown voxels that touch visible space. See the header note.
    bool     solid      = false;

    // See carve::EarlyOut. Saturated is exact; AnyEvidence is exact for the
    // unknown set only and has to be asked for.
    carve::EarlyOut earlyOut = carve::EarlyOut::Saturated;

    DomainMode domain = DomainMode::MeasuredExtent;
    // How far past the last measured return the question still applies. Two
    // metres covers wall thickness, eaves, and registration slop — enough that a
    // void just behind a surface is still asked about, without reaching into the
    // open air the scan was never about. It is a physical depth, not a tolerance,
    // so it is in metres rather than derived from the voxel size.
    double   domainMargin = 2.0;

    // Passed through to rimg::Options. OFF (radius 0), matching rimg's own
    // default: a ray either returned or it did not, and one that did not is not
    // second-guessed from the shape of the empty region around it. These two
    // defaults have to agree — they disagreed once, and the effect was that the
    // library said the filter was off while every run through the CLI and the
    // app had it on.
    uint32_t skyRadius   = 0;
    double   skyFraction = 0.75;

    // Which end of each raster holds the instrument's blind cone. Auto finds it
    // from the geometry and copes with a scanner mounted upside down.
    rimg::BlindCone blindCone = rimg::BlindCone::Auto;

    // Separate voids from the rest of the world — see voids.h.
    //
    // OFF by default. It answers a narrower question than the one usually being
    // asked: it keeps only unobserved space you cannot reach from outside
    // without crossing observed space, which excludes a building interior whose
    // walls were only ever seen from one side. That is often exactly the space
    // you wanted reported, so this hides more than it helps until the enclosure
    // it depends on is actually there.
    //
    // Useful when the site is genuinely enclosed and the question is "what did I
    // miss inside it". Needs one byte per voxel of the whole domain at once,
    // because connectivity cannot be answered tile by tile.
    bool     classifyVoids = false;
    uint64_t classifyBudgetBytes = 6ull << 30;

    uint64_t totalImageCells = 512ull << 20;
    uint32_t minImageCells   = 1u << 20;   // never bin below this per scan

    // Upper bound on voxels handed back for drawing. 6 M is 120 MB as
    // StorePoints, which is a fraction of what the point cloud itself costs.
    uint64_t displayCap = 6ull << 20;

    // An accelerator for carveTile — the Metal path installs itself here. Left
    // null the carve runs on the CPU, which is what happens anyway whenever the
    // carver declines a tile. A carver is driven from one thread: it is the
    // parallelism, so the tile pool would only contend with it.
    carve::TileCarver carver = nullptr;
    void*             carverUser = nullptr;
    // Carve every tile both ways and count the voxels they disagree about.
    // Halves the speed, obviously; it is how you find out whether a new carver
    // is telling the truth on real data rather than on a fixture.
    bool              verifyCarver = false;

    // Worker threads over the tile list. 0 asks the machine. Tiles share
    // nothing — no accumulator, no neighbour reads across seams — so this is
    // parallel by construction rather than by locking, and the result is
    // identical at any thread count: the statistics are integer sums, and the
    // display sample is chosen by a hash threshold rather than by counting.
    uint32_t threads = 0;
};

// done/total are 0 when a stage cannot say. Return false to cancel; a cancelled
// run reports what it had and sets `cancelled`.
using Progress = std::function<bool(const std::string& stage, uint64_t done, uint64_t total)>;

struct Result {
    carve::Stats stats;
    uint64_t     tilesCarved = 0;
    uint64_t     tilesTotal  = 0;
    uint64_t     setupsUsed  = 0;
    uint64_t     scansSkipped = 0;
    // Setups whose angular mapping was refused. They are in the corpus and in
    // the setup list, and they contribute nothing at all: every lookup against
    // them is outside the raster. Counted because a scan that silently says
    // nothing looks exactly like a scan that saw nothing, and the difference is
    // the whole answer.
    uint64_t     setupsWithoutMapping = 0;
    // Setups whose blind cone was found, and how many were mounted inverted.
    uint64_t     setupsWithBlindCone = 0;
    uint64_t     setupsInverted = 0;
    // How the cone was decided across the whole corpus, which is a far stronger
    // signal than any single scan affords — see rimg::markBlindConeAcrossCorpus.
    rimg::ConeVerdict coneVerdict;
    bool         partial   = false;    // maxTiles stopped it short
    bool         cancelled = false;

    // Tiles the installed carver accepted, and what verification found.
    uint64_t carverTiles = 0;
    uint64_t carverRefused = 0;
    uint64_t carverDisagreements = 0;
    uint64_t carverVoxelsCompared = 0;

    // What the connectivity pass found, when it ran.
    bool         classified = false;
    voids::Report voidReport;
    std::string  classifySkipped;      // why not, when it did not run

    // The region actually asked about, and how much smaller it made the job.
    carve::Domain domain;
    double        domainVolume = 0;      // m^3 of the domain box, 0 when unbounded
    double        sphereVolume = 0;      // m^3 of the range-sphere bounding box

    // Voxels to draw, as StorePoints so the existing point pipeline can render
    // them with no new shader. Positions are metres relative to `origin`.
    std::vector<lod::StorePoint> voxels;
    double   origin[3] = {0, 0, 0};
    double   voxelSize = 0;
    lod::Aabb bounds;

    // How many voxels qualified before the cap, and the sampling that was
    // applied to fit. `keptFraction` is 1.0 when nothing was dropped.
    uint64_t qualified = 0;
    double   keptFraction = 1.0;

    std::string note;

    double voxelVolume() const { return voxelSize * voxelSize * voxelSize; }
    double unknownVolume() const { return double(stats.unknown) * voxelVolume(); }
    // The deliverable when the classification ran: space enclosed by what the
    // scanners saw, and therefore genuinely missed rather than merely elsewhere.
    double enclosedVolume() const { return double(voidReport.enclosed) * voxelVolume(); }
    double exteriorVolume() const { return double(voidReport.exterior) * voxelVolume(); }
};

// The whole job: open each file, build a range image per scan, carve, reduce.
// Returns false only when it could not start at all — a corpus with no usable
// structured scan, or an unreadable file.
bool run(const std::vector<std::string>& paths, const Options& opt,
         const Progress& progress, Result& out, std::string& err);

// Re-expresses the voxels against a different origin, which is what the viewer
// needs: the point store picks its own origin and the two have to agree or the
// voxels float away from the cloud.
void rebase(const Result& r, const double origin[3], std::vector<lod::StorePoint>& out);

// Exposed for testing: the frontier rule and the sampling decision.
bool touchesVisible(const carve::Tile& t, uint32_t x, uint32_t y, uint32_t z);
uint64_t voxelHash(int64_t x, int64_t y, int64_t z);

} // namespace vis
