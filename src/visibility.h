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
#include "wrap.h"

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
    // A shrinkwrap of the returns: the space within `domainMargin` of something
    // a scanner actually measured. See wrap.h — the box's corners that no scan
    // reached come out of the question, which is both the largest speed factor
    // available and what makes the unobserved volume mean "coverage stops here"
    // rather than "mostly sky".
    Shrinkwrap,
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

    // Shrinkwrap settings; ignored unless `domain` is Shrinkwrap. The buffer is
    // `domainMargin` — the same parameter, meaning the same thing, applied to a
    // wrap instead of a box.
    //
    // `wrapInteriorOnly` is the one switch that distinguishes the two kinds of
    // survey. False for one that looks outward, where the shadow behind a wall is
    // part of the answer and the buffer is what stops it running to the horizon.
    // True for one conducted entirely inside a building, where the space outside
    // the walls is not the question — and where a shell of unobserved voxels
    // wrapped round the outside hides everything within it.
    //
    // It is a switch rather than something inferred, because getting it wrong
    // silently would either hide a building's interior or hide nothing at all,
    // and neither announces itself in the picture.
    bool     wrapInteriorOnly = false;
    double   wrapCell     = 0.0;             // 0 derives it from the buffer
    uint64_t wrapMaxCells = 64ull << 20;
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

    // How much memory the range images may occupy, every scan at once.
    //
    // They are the largest thing the filter holds and the only part of it that
    // grows with the number of scans, so this is the number that decides whether
    // a corpus fits. 48 GB, for the 64 GB machines this is built for.
    //
    // In bytes rather than cells because bytes are what runs out, and because
    // what a cell costs is an implementation detail — it is three bytes of
    // raster plus about a third of a byte of pyramid over it, and vis::run does
    // the conversion.
    //
    // Sized from what a real corpus needs: a 2500 x 5280 terrestrial raster is
    // 13.2 M cells, about 44 MB with its pyramid, and a thousand of them is
    // 44 GB. So a thousand-scan job runs at full resolution with a little room
    // left, which is the case this is for.
    //
    // What happens when a corpus does NOT fit is why this is generous rather
    // than cautious. The budget is divided by the number of scans and each
    // raster is binned down to fit its share — and binning does not blur the
    // answer, it changes it. A coarse cell keeps the nearest of the several
    // returns that land in it, because line of sight stops at the first surface,
    // so a coarse raster clears less space and space that WAS observed starts
    // reporting as unobserved. Measured on a 36-setup site: squeezing the images
    // to 1 M cells each took the unobserved volume from zero voxels to 96.8
    // million, and made the carve three times slower into the bargain. The
    // previous 512 M cell budget would have given each of a thousand rasters
    // 512 k cells — half that again.
    //
    // Which is a silent wrong answer, so it is no longer silent: Result reports
    // how many scans were binned and by how much. Set this lower deliberately,
    // never by accident.
    uint64_t imageBudgetBytes = 48ull << 30;
    uint32_t minImageCells    = 1u << 20;   // never bin below this per scan

    // How the drawn voxels are coloured to begin with. 0 flat, 1 lit, 2 height
    // ramp, 3 both. Changeable afterwards without re-carving — see recolour.
    //
    // Both by default. A frontier drawn in one flat colour is a silhouette with
    // no interior — you can see where the unobserved volume is and nothing about
    // its shape — and the two cues that fix it are free: the frontier test
    // already looked at the six face neighbours, which is an outward normal, and
    // the domain's height range is known before any voxel is collected. See
    // shadeFrontier in visibility.cpp.
    //
    // A plain integer rather than an enum because it crosses to the app's
    // options struct and back; the values are vis::Shade.
    uint8_t  shading = 3;

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
    // Whether the images fitted the memory budget at full resolution, and what
    // was given up if not. `setupsBinned` counts rasters that had to be coarsened
    // and `worstBinStep` is how far the worst of them went, in declared grid cells
    // per raster cell per edge.
    //
    // Reported because binning is not a loss of sharpness, it is a change of
    // answer in one direction: coarse cells clear less space, so observed space
    // reports as unobserved and the unknown volume is overstated. A run that did
    // this quietly would hand back a number nobody could interpret. See
    // Options::imageBudgetBytes.
    uint64_t     imageCellsAllowed = 0;
    uint64_t     setupsBinned = 0;
    uint32_t     worstBinStep = 1;
    // Setups whose angular mapping was refused. They are in the corpus and in
    // the setup list, and they contribute nothing at all: every lookup against
    // them is outside the raster. Counted because a scan that silently says
    // nothing looks exactly like a scan that saw nothing, and the difference is
    // the whole answer.
    uint64_t     setupsWithoutMapping = 0;
    // Why the first of them was refused, verbatim. "Refused" on its own is not
    // something anyone can act on, and a run that produces no voxels at all should
    // say what stopped it without needing a second command run afterwards.
    std::string  mappingRefusedWhy;
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
    // Which of each drawn voxel's six face neighbours were observed, in the
    // order vis::kFaceDirs lists them — its outward normal, kept so the shading
    // can be changed without carving the site again. One byte a voxel, six
    // megabytes at the display cap, against a carve that takes minutes.
    //
    // Parallel to `voxels` and the same length: everything that filters or
    // reorders one does the same to the other.
    std::vector<uint8_t> voxelFaces;
    // The wrap, when one was built. Owned here because `domain` points into it:
    // a Result that outlives the run has to carry the grid its own domain refers
    // to, or the predicate is left pointing at a dead stack frame.
    wrap::Grid  wrapGrid;
    // Why there is no wrap, when one was asked for and could not be had.
    std::string wrapNote;
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

// How many cells each scan's raster may occupy, given the memory budget and how
// many scans have to share it. What run() uses, exposed so the sizing can be
// checked against a real raster without reading a corpus to do it: a terrestrial
// 2500 x 5280 raster is 13.2 M cells, and whether a thousand of them fit is a
// question about this function and nothing else.
uint64_t imageCellsPerScan(const Options& opt, uint64_t scanCount);

// Recolours a finished result in place — see Options::shading for the modes.
//
// Separate from run() because shading is a way of looking at the answer, not
// part of computing it, and the two should not share a cost. Everything it needs
// was kept: the outward normal in `voxelFaces`, and the height range in
// `domain`. A viewer can offer the modes as a menu and switch between them on a
// finished carve, instead of asking for the site to be carved again to change a
// colour.
void recolour(Result& r, uint8_t shading);

// Exposed for testing: the frontier rule and the sampling decision.
bool touchesVisible(const carve::Tile& t, uint32_t x, uint32_t y, uint32_t z);
uint64_t voxelHash(int64_t x, int64_t y, int64_t z);

} // namespace vis
