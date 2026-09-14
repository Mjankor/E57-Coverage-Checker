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
//   that is neither possible nor useful, so there is a display cap and a
//   reduction: the frontier, meaning unknown voxels with a face neighbour that
//   was OBSERVED — seen through, or measured on. That is where coverage stops:
//   the mouth of a shadow, the far edge of the range spheres, the back of the
//   ceiling a beam stopped on.
//
//   The reduction is OFF by default now, and the reason is worth keeping. It
//   rests on a solid body looking the same as its own surface from outside, and
//   that fails wherever the body's boundary was never observed either. The blind
//   cone under a single setup is exactly that: the floor inside the cone is never
//   measured, so the unobserved space beneath it touches nothing observed, gets
//   dropped, and from below the cone reads as a hole carved through the answer.
//   Nothing was carved — the volume is counted as unobserved throughout — but a
//   reduction that can hide a whole unobserved region is not a safe default.
//
//   "Observed" rather than "visible" is a correction in the same direction: the
//   slab of unobserved space above a ceiling is bounded by measured surface below
//   and by itself everywhere else, so asking for a neighbour seen THROUGH left
//   the whole slab undrawable and an indoor survey showed nothing above its roof.
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

    // Draw every unobserved voxel, not only the ones touching observed space.
    //
    // ON by default, and that is a correction. The frontier reduction is a claim
    // about what a solid body looks like from outside, and the claim does not hold
    // where the body's boundary was never observed either. The blind cone under a
    // single setup is the case: the patch of floor inside it is never measured, so
    // the unobserved space below that patch touches nothing observed, is dropped,
    // and the cone reads from beneath as a hole carved through the middle of the
    // answer. It is not carved — every voxel of it is unobserved and counted as
    // such. Reducing it away is the reduction being wrong about what is safe to
    // hide, so the reduction is no longer the default.
    //
    // Set it false to get the frontier back: it is still much the cheaper thing to
    // draw, and on a site where every surface is measured from several setups the
    // two look the same from outside.
    bool     solid      = true;

    // See carve::EarlyOut. Saturated is exact; AnyEvidence is exact for the
    // unknown set only, which is the set being reported, and it is much faster —
    // so it is the default and `visible` and `occupied` are read as lower bounds.
    // Means nothing under Method::RayMarch, which has no per-voxel loop to stop.
    carve::EarlyOut earlyOut = carve::EarlyOut::AnyEvidence;

    // INDOOR OR OUTDOOR, SAID BY THE OPERATOR RATHER THAN THE SKY TEST.
    //
    // One entry per scan the operator has an opinion about, keyed the way the
    // setups table keys its rows: the file it came from and the scan's index
    // within it. Scans with no entry fall through to the test, so a corpus where
    // two stations were marked by hand still has its other nine hundred decided
    // from the data.
    //
    // Only consulted when `useSkyOverrides` is set — the run sheet's switch between
    // "the sky test decides" and "the column decides". Kept separate because the
    // marks are worth remembering across runs while a sweep is being done with the
    // test, and a switch is how you compare the two without losing the marks.
    struct SkyOverride {
        std::string path;
        uint32_t    scanIndex = 0;
        bool        outdoor   = false;
    };
    std::vector<SkyOverride> skyOverrides;
    bool useSkyOverrides = false;

    // Which formulation of the carve to run — see carve::Method. RayMarch is the
    // specified one and the default; the two gathers are kept so the three can be
    // run against each other on real data.
    //
    // Taken from carve::Params rather than written out, and NOT `carve::Method{}`:
    // that default-constructs to the first enumerator, which is CentreRay — the one
    // that leaves a third of a room unobserved — so the app would have shipped
    // running the method the other two exist to replace, while every test that built
    // a carve::Params directly got RayMarch and agreed with itself.
    carve::Method method = carve::Params{}.method;

    // The tightest region, and the one that makes the fraction mean something.
    DomainMode domain = DomainMode::Shrinkwrap;

    // Shrinkwrap settings; ignored unless `domain` is Shrinkwrap. The buffer is
    // `domainMargin` — the same parameter, meaning the same thing, applied to a
    // wrap instead of a box, and SIGNED: see wrap::Options::buffer.
    //
    // There is no interior-only switch any more. It said "leave out the space
    // past the walls", which a negative margin says better: as a distance, in the
    // units the rest of the sheet already uses, and by moving the boundary inside
    // the wall rather than by hugging both of its faces.
    //
    // `wrapSpanGaps` is how wide an opening the shell may bridge — see
    // wrap::Options::spanGaps. Without it a wrap dips into every window reveal
    // and runs through every open door, which threads it into the rooms behind.
    // Two metres: a door, a window, and the gaps a raster leaves in a wall at a
    // grazing angle are all narrower than that, and a lane between two buildings
    // is wider. On by default because the negative buffer depends on it — that
    // question asks which side of the shell a cell is on, and an envelope with a
    // hole in it has no sides.
    double   wrapSpanGaps = 2.0;
    double   wrapCell     = 0.0;             // 0 derives it from the buffer
    uint64_t wrapMaxCells = 64ull << 20;
    // How far past the last measured return the question still applies. Two
    // metres covers wall thickness, eaves, and registration slop — enough that a
    // void just behind a surface is still asked about, without reaching into the
    // open air the scan was never about. It is a physical depth, not a tolerance,
    // so it is in metres rather than derived from the voxel size.
    // How far past the last return the question still applies, in metres. The
    // box's margin and the shrinkwrap's buffer, which are one thing.
    //
    // MAY BE NEGATIVE, and the two regions honour that differently because the
    // shapes differ:
    //
    //   Box. The box shrinks. On an indoor job the extent hugs the building, so a
    //   positive margin asks about a couple of metres past every wall where
    //   nothing could ever be seen; a negative one pulls the question inside the
    //   walls and that space is never asked about. Clamped at half the site on
    //   each axis so the box cannot invert — an inverted box would carve nothing
    //   and report perfect coverage.
    //
    //   Shrinkwrap. The sign goes straight through — see wrap::Options::buffer.
    //   Positive is a skin around the measured surfaces; negative is the region
    //   the survey encloses, pulled in by that much, so the boundary sits inside
    //   the outer face of the walls and nothing beyond them is in the question.
    //
    // Half a metre by default, which is a tight question: it asks about the space
    // close to what was measured and nothing else.
    double   domainMargin = -0.2;

    // The instrument's rated MINIMUM range, in metres, passed through to
    // rimg::Options. A surface inside it returns nothing, and believing that
    // no-return clears a pencil of space straight through the surface — see
    // rimg::filterNoReturnsTooClose. 0 switches the test off.
    double   minRange = rimg::Options{}.minRange;

    // How wide the opening at a scan's own zenith has to be before it is called
    // sky, in degrees, passed through to rimg::Options. Surfaced on the run sheet
    // beside the minimum range because both are judgements about what an empty
    // cell means, and both want trying against real data: this one decides how
    // much of a roofless or open-sided scene clears. See rimg::identifySky.
    double   skyMinExtentDeg = rimg::Options{}.skyMinExtentDeg;

    // And over how wide an arc of bearing that angle has to be reached, in degrees,
    // passed through to rimg::Options. Surfaced beside the angle because it is the
    // other half of the same judgement: the angle says how far the opening must
    // reach, this says how far around it must reach that far. 0 leaves the
    // single-angle test, which a dead strip up a door frame satisfies over about a
    // degree of bearing. See rimg::Options::skyMinArcDeg.
    double   skyMinArcDeg = rimg::Options{}.skyMinArcDeg;

    // What share of a no-return zone's bordering returns has to be near the
    // bottom of this scan's own intensity distribution before the zone is
    // disbelieved, passed through to rimg::Options. 0 switches the test off and 1
    // demotes almost nothing. See rimg::filterDarkBorderedZones.
    double   darkBorderFraction = rimg::Options{}.darkBorderFraction;

    // Only a no-return the scan named as its own sky clears space; every other
    // empty cell establishes nothing. ON by default — see rimg::Options::skyOnly,
    // which carries the reasoning and the cost.
    bool     skyOnly = rimg::Options{}.skyOnly;

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
    // Voxels DRAWN at most. Over this the frontier is sampled by position hash.
    //
    // 24 M is about 480 MB of StorePoint, which is nothing on a machine carving a
    // corpus this size, and it is what the default had to become: a large site at
    // 5 cm has tens of millions of frontier voxels, and the old 6 M scattered
    // through a whole building drew as almost nothing — indistinguishable from a
    // run that found no unobserved space. Result::keptFraction says when it bit,
    // and both the CLI and the run sheet now report it.
    uint64_t displayCap = 24ull << 20;

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
    // WHETHER EACH SETUP SAW THE SKY, one entry per scan that produced a usable
    // range image, in the order they were built.
    //
    // The aggregate counts below say how many; this says which, and on what
    // evidence, because "three of your setups are outdoors" is not something anyone
    // can check and "this one, on an opening 44 degrees wide right around, bordered
    // at three metres" is. It is what fills the setups table's own column, and what
    // an operator disagrees with by marking that column.
    struct SetupSky {
        std::string path;                 // the file, as the table keys its rows
        uint32_t    scanIndex = 0;
        bool        outdoor = false;      // the sky was named and believed
        bool        reachedPole = false;  // there was an opening at the zenith at all
        bool        overridden = false;   // the verdict came from a mark, not the test
        double      extentDeg = 0;        // and the three numbers behind it
        double      arcDeg = 0;
        double      borderM = -1;
    };
    std::vector<SetupSky> setupSky;

    // Setups whose blind cone was found, and how many were mounted inverted.
    uint64_t     setupsWithBlindCone = 0;
    // Setups that had at least one zone of no-returns inside the instrument's
    // minimum range, and how many cells those zones held in total. Each of those
    // cells would otherwise have cleared a pencil of space to the rated range
    // through the surface that was too close to measure — see
    // rimg::filterNoReturnsTooClose. The nearest bordering range any of them was
    // judged on is kept as well, in metres, so the figure the decision turned on
    // is visible rather than inferred.
    uint64_t     setupsTooClose = 0;
    // What is still BELIEVED once the two instrument cases are accounted for:
    // every one of these cells clears a ray to the rated range, and if any of them
    // is wrong this is where it is. `believedCells` is the total; the rest describe
    // the largest single zone found in any scan, which is the one worth looking at
    // first — see rimg::describeNoReturnZones for the same breakdown per scan.
    uint64_t     believedCells = 0;
    // Setups where the sky could be named — an opening at the instrument's own
    // zenith wider than a cone about it — and setups with zones whose bordering
    // returns were too weak to believe. See rimg::identifySky and
    // rimg::filterDarkBorderedZones.
    uint64_t     setupsWithSky = 0;
    // And setups where the opening at the zenith fell SHORT of the angle, so it is
    // a hole in whatever the instrument was under and clears nothing. The active
    // half of the sky check on an indoor job, where there is no sky to find.
    uint64_t     setupsZenithClosed = 0;
    uint64_t     zenithDemotedCells = 0;
    uint64_t     setupsWithDarkZones = 0;
    uint64_t     darkCells = 0;
    uint64_t     setupsWithoutIntensity = 0;
    uint64_t     largestZoneCells = 0;
    double       largestZoneBorderMinM = -1.0;
    double       largestZoneBorderMedianM = -1.0;
    double       largestZoneElLoDeg = 0.0, largestZoneElHiDeg = 0.0;
    uint64_t     tooCloseCells  = 0;
    double       tooCloseNearest = -1.0;
    // Setups with an unsampled band at an end that was left BELIEVED — treated as
    // rays that saw through to the rated range rather than as directions never
    // sampled. Each one clears a cone through whatever is beyond it, and because
    // the elevation table is extrapolated across the band, that cone is aimed
    // straight at the pole: up through a roof, down through a floor.
    //
    // Reported per end, because which end it is says what is being lost.
    uint64_t     setupsBandBelievedLow  = 0;
    uint64_t     setupsBandBelievedHigh = 0;
    uint64_t     setupsInverted = 0;
    // What the scans decided about their own blind cones, tallied up — each one
    // decides from its own raster, see rimg::markBlindCone.
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
    // And the part of it the answer is REPORTED over, which the buffer's sign
    // chooses out of the union the carve was asked about. Equal to domainVolume
    // where the region is not a wrap.
    double        reportedVolume = 0;
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
    // The wrap, as something to look at. See buildWrapSkin: the boundary of the
    // domain, plus the cells that hold returns, in one drawable set against the
    // same `origin` as the voxels.
    //
    // A picture rather than a statistic, and it exists because the wrap decides
    // what the whole answer covers while being invisible in that answer: a wrap
    // that has gone wrong looks, in the voxels, exactly like a survey that
    // missed different space.
    std::vector<lod::StorePoint> wrapSkin;
    uint64_t wrapSkinCells = 0;      // before the display cap
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

// GETTING THE ANSWER BACK OUT.
//
// Which of a run's two point sets to write. The unobserved voxels are the
// deliverable; the shell is what decided the question they were asked inside, and
// a run where the shell went wrong looks, in the voxels alone, exactly like a
// survey that missed different space — so it has to be saveable too.
enum class SavePart {
    UnobservedVoxels,
    Shrinkwrap,
};

// Writes one of them as a PLY, with a header describing the run that produced it
// — the build, the settings that decide what an empty cell means, the voxel size,
// the counts, and whether the set was sampled to fit the display cap.
//
// The saved set is what the run produced, which for the voxels is what the cap
// let through: a run that sampled draws a sample and saves the same sample, and
// the header says so in as many words rather than handing over a thinned cloud
// that looks complete. Raise "voxels drawn at most" to save the lot.
//
// False with a reason when there is nothing to write or the file cannot be made.
bool save(const Result& r, const Options& opt, SavePart part,
          const std::string& path, std::string& err);

// The one-line summary of what a save wrote, for a status line.
std::string saveSummary(const Result& r, SavePart part, const std::string& path);

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

// Builds Result::wrapSkin from Result::wrapGrid. Called by run(); exposed so a
// caller holding a result can rebuild it, and so it can be tested.
void buildWrapSkin(Result& r, uint64_t cap);

// Keeps only the unknown voxels that lie INSIDE the wrap, dropping the rest —
// the boolean intersection of the answer with the shrinkwrap volume.
//
// Containment, not proximity to the shell. That distinction is the whole point and
// it is easy to get backwards: what is wanted is everything within the wrap, not
// the thin layer at its edge.
//
// What it does to a wall, which is the case worth holding in mind. On the scanner
// side, the scanner saw the wall, so that space is visible and there are no
// unknown voxels to keep — the intersection correctly picks up nothing there. On
// the far side the scanner saw nothing, so unknown voxels run from the wall out to
// the edge of range; the wrap reaches only a buffer past the wall, so what
// survives is one buffer's thickness of them against the back of the wall. That
// is the statement worth drawing: coverage stopped HERE. The rest of that column,
// running to the range limit, is space nobody ever asked about and it is what
// turns the display into a solid mass.
//
// Filters Result::voxels and Result::voxelFaces together, so the shading stays
// attached to the voxel it was computed for. Returns how many were kept.
// Destructive: the dropped voxels are gone from the Result, so a caller wanting
// both views keeps a copy. A Result with no wrap grid keeps everything, there
// being nothing to intersect with.
//
// Most useful on a run whose DOMAIN was wider than the wrap — a box or the range
// spheres — since a run already carved over the wrap has its voxels inside it
// by construction and this then changes nothing.
uint64_t keepVoxelsInsideWrap(Result& r);

// Exposed for testing: the frontier rule and the sampling decision.
bool touchesObserved(const carve::Tile& t, uint32_t x, uint32_t y, uint32_t z);
uint64_t voxelHash(int64_t x, int64_t y, int64_t z);

} // namespace vis
