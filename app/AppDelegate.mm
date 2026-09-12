// Window, scan list, and the two-phase open.
//
// Opening a corpus happens in two stages, because at a thousand files the two
// have wildly different costs:
//
//   1. Survey — headers only. Pose, prototype, declared extent, metadata
//      classification. No point decoding, so a thousand files take seconds.
//      The setup layout is drawn immediately and the list is populated, which
//      is enough to see the shape of the job and spot a stray setup.
//
//   2. Index — the octree build. Minutes on a large corpus, and cached: a
//      store is keyed by the file list and their sizes and modification times,
//      so reopening the same corpus skips straight to stage two's result.
//
// Waiting for stage 2 before showing anything would mean a blank window for
// minutes. Doing stage 2 in the foreground would mean a beachball.

#import <Cocoa/Cocoa.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

#import "CarveGpu.h"
#import "CloudView.h"

#include "../src/indexer.h"
#include "../src/point_store.h"
#include "../src/scan_check.h"
#include "../src/report.h"
#include "../src/version.h"
#include "../src/visibility.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

// Above this share of voxels, a carver's disagreement with the CPU is a fault
// rather than arithmetic.
//
// The Metal kernel computes in float and the CPU in double, and the two
// genuinely differ for a voxel whose direction lands within a rounding error of
// a raster cell boundary — the bin index runs to tens of thousands, where float
// resolves about a four-hundredth of a bin. Replaying both paths over a real
// 5.6 M point scan put 0.014% of voxels in a different cell, and every single
// disagreement sat within 0.006 of a bin of an edge. Metal has no float64, so
// that floor cannot be lowered.
//
// 0.1% is five times the measured ceiling and two orders below what a real
// kernel fault produces — a wrong index or a wrong transform misses by whole
// cells, not by a thousandth of one. Wide enough not to cry wolf, narrow enough
// that nothing real hides under it.
constexpr double kCarverFloatNoise = 0.001;

NSString *ns(const std::string &s) { return [NSString stringWithUTF8String:s.c_str()]; }

std::string humanCount(uint64_t n) {
    char buf[64];
    if (n >= 1000000000ull) std::snprintf(buf, sizeof(buf), "%.2f G", double(n) / 1e9);
    else if (n >= 1000000ull) std::snprintf(buf, sizeof(buf), "%.1f M", double(n) / 1e6);
    else if (n >= 1000ull) std::snprintf(buf, sizeof(buf), "%.1f k", double(n) / 1e3);
    else std::snprintf(buf, sizeof(buf), "%llu", (unsigned long long)n);
    return buf;
}

// Identifies a corpus by its file list plus each file's size and mtime, so an
// edited or replaced scan invalidates the cached store rather than silently
// showing stale geometry.
std::string corpusKey(const std::vector<std::string> &paths) {
    std::vector<std::string> sorted = paths;
    std::sort(sorted.begin(), sorted.end());
    uint64_t h = 1469598103934665603ull;              // FNV-1a
    auto mix = [&h](const void *data, size_t n) {
        const uint8_t *p = static_cast<const uint8_t *>(data);
        for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    };
    for (const auto &p : sorted) {
        mix(p.data(), p.size());
        struct stat st{};
        if (::stat(p.c_str(), &st) == 0) {
            const uint64_t size = uint64_t(st.st_size);
            const uint64_t time = uint64_t(st.st_mtime);
            mix(&size, sizeof(size));
            mix(&time, sizeof(time));
        }
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
    return buf;
}

// The sidebar's default width. Narrow on purpose — it identifies setups, and the
// cloud is what the window is for — but wide enough that all three columns fit
// with the scroller: 150 + 104 + 60 of column, two 3 px gaps, 16 for the
// scroller, and a little slack.
constexpr CGFloat kSidebarWidth    = 344;
// Below this the three columns cannot all meet their minimum widths
// (80 + 70 + 52, two 3 px gaps, and the scroller).
constexpr CGFloat kSidebarMinWidth = 240;
constexpr CGFloat kSidebarMaxWidth = 620;
// The cloud never gets squeezed to nothing, however far the divider is dragged.
constexpr CGFloat kCloudMinWidth   = 360;
constexpr CGFloat kDividerWidth    = 1;
// Wider than the line it sits over: a one-pixel drag target is unusable.
constexpr CGFloat kGripWidth       = 9;
// The strip under the cloud holding the status and progress lines.
constexpr CGFloat kStatusStripHeight = 44;
// Space left around the window on the desktop. Not zero, so it still reads as a
// window rather than as a takeover, and so the corners stay grabbable.
constexpr CGFloat kScreenInset     = 20;

const char *kindLabel(check::Kind k) {
    switch (k) {
    case check::Kind::Structured: return "structured";
    // Not "not indexed" any more, because it is indexed. The label says what was
    // observed and leaves the conclusion alone: the measurement behind it rises
    // with scene scale, so an ordinary outdoor setup reads this way. See the note
    // at the top of scan_check.h.
    case check::Kind::Unified:    return "no grid declared, spread suggests several";
    case check::Kind::Ambiguous:  return "ambiguous";
    }
    return "?";
}

} // namespace

// The strip between the list and the cloud. A one-pixel line reads correctly but
// is impossible to grab, so the line and the grab area are separate views: a
// hairline that is drawn, and a wider transparent one over it that is dragged.
@class AppDelegate;

@interface DividerGrip : NSView
@property (nonatomic, weak) AppDelegate *owner;
@end

@interface AppDelegate : NSObject <NSApplicationDelegate, NSTableViewDataSource,
                                   NSTableViewDelegate, NSWindowDelegate,
                                   CloudViewDelegate>
@end

@implementation AppDelegate {
    NSWindow            *_window;
    CloudView           *_cloudView;
    NSTableView         *_table;
    NSTextField         *_status;
    NSTextField         *_progress;
    NSProgressIndicator *_spinner;

    NSScrollView        *_scroll;
    NSView              *_divider;
    DividerGrip         *_grip;
    NSView              *_rightPane;
    CGFloat              _sidebarWidth;
    NSButton            *_cloudToggle;
    NSButton            *_voxelToggle;
    NSButton            *_wrapToggle;
    NSMutableArray<NSMenuItem *> *_shadingItems;
    // Whether to draw only the unobserved voxels inside the shrinkwrap. A display
    // filter over a finished carve, remembered between runs like the rest of the
    // run sheet's settings.
    BOOL                 _intersectWrap;
    NSWindow            *_reportWindow;
    NSTextView          *_reportText;

    indexer::Survey      _survey;
    // The corpus as opened, so the visibility pass can be run over exactly the
    // files the view is showing.
    std::vector<std::string> _paths;
    // Carried between runs so the sheet reopens with what was last used.
    vis::Options         _visOptions;
    BOOL                 _useGpu;
    BOOL                 _verifyGpu;
    BOOL                 _busy;
    // Shared with the worker rather than read through `self`: it is written on
    // the main thread and read on a background queue, which as a plain BOOL was
    // a data race.
    std::shared_ptr<std::atomic<bool>> _cancel;
}

- (void)applicationDidFinishLaunching:(NSNotification *)note {
    (void)note;
    [self buildMenu];

    // The working area, less a small inset. No absolute cap: a point cloud is
    // read by eye and there is no display large enough that you would rather
    // have grey desktop beside it.
    NSScreen *screen = NSScreen.mainScreen;
    const NSRect visible = screen ? screen.visibleFrame : NSMakeRect(0, 0, 1440, 900);
    const NSRect frame = NSMakeRect(0, 0,
                                    std::max(900.0, visible.size.width  - 2 * kScreenInset),
                                    std::max(600.0, visible.size.height - 2 * kScreenInset));
    _window = [[NSWindow alloc]
        initWithContentRect:frame
                  styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                             NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable)
                    backing:NSBackingStoreBuffered
                      defer:NO];
    _window.title = @"E57 Coverage Checker";
    _window.contentMinSize = NSMakeSize(900, 600);
    _window.delegate = self;
    [_window center];

    // A plain container, laid out by hand in -layoutPanes.
    //
    // This was an NSSplitView twice, and twice the cloud stopped short of the
    // window's right edge with a band of empty background beside it. Split views
    // manage their subviews' frames through a delegate protocol that interacts
    // with autoresizing masks, and getting that combination wrong is invisible
    // until you see the window. The layout here is thirty lines and reads its
    // size from contentView.bounds, which is by definition the window's content
    // size. Nothing to get subtly wrong, and nothing that needs a Mac to check.
    NSView *root = [[NSView alloc] initWithFrame:frame];
    root.autoresizesSubviews = NO;

    _table = [[NSTableView alloc] initWithFrame:NSZeroRect];
    _table.dataSource = self;
    _table.delegate = self;
    _table.usesAlternatingRowBackgroundColors = YES;
    _table.rowHeight = 30;
    // Widths are set in -layoutPanes from the width actually available, so all
    // three columns are visible whatever the sidebar has been dragged to. These
    // are only the starting proportions.
    struct { NSString *ident; NSString *title; CGFloat minWidth; BOOL right; } cols[] = {
        {@"scan",   @"Setup",  80, NO},
        {@"status", @"Status", 70, NO},
        {@"points", @"Points", 56, YES},
    };
    for (auto &c : cols) {
        NSTableColumn *col = [[NSTableColumn alloc] initWithIdentifier:c.ident];
        col.title    = c.title;
        col.minWidth = c.minWidth;
        col.width    = c.minWidth;
        if (c.right) col.headerCell.alignment = NSTextAlignmentRight;
        [_table addTableColumn:col];
    }
    // The columns are sized explicitly, so the table must not resize them back.
    _table.columnAutoresizingStyle = NSTableViewNoColumnAutoresizing;

    _scroll = [[NSScrollView alloc] initWithFrame:NSZeroRect];
    _scroll.documentView = _table;
    _scroll.hasVerticalScroller = YES;
    // An overlay scroller would float over the last column; a legacy one takes
    // its own width, which -layoutPanes subtracts before sizing the columns.
    _scroll.scrollerStyle = NSScrollerStyleLegacy;
    _scroll.borderType = NSNoBorder;
    [root addSubview:_scroll];

    _divider = [[NSView alloc] initWithFrame:NSZeroRect];
    _divider.wantsLayer = YES;
    _divider.layer.backgroundColor = NSColor.separatorColor.CGColor;
    [root addSubview:_divider];

    _rightPane = [[NSView alloc] initWithFrame:NSZeroRect];
    _rightPane.autoresizesSubviews = NO;
    [root addSubview:_rightPane];

    _cloudView = [[CloudView alloc] initWithFrame:NSZeroRect];
    _cloudView.cloudDelegate = self;
    [_rightPane addSubview:_cloudView];

    _status = [[NSTextField alloc] initWithFrame:NSZeroRect];
    _progress = [[NSTextField alloc] initWithFrame:NSZeroRect];
    for (NSTextField *f in @[_status, _progress]) {
        f.bezeled = NO; f.editable = NO; f.drawsBackground = NO;
        f.font = [NSFont monospacedDigitSystemFontOfSize:11 weight:NSFontWeightRegular];
        f.textColor = [NSColor secondaryLabelColor];
        [_rightPane addSubview:f];
    }
    // The build, on screen from the first frame. Two rounds of diagnosis were
    // spent on a stale binary that looked identical to a current one.
    _status.stringValue = [NSString stringWithFormat:
        @"build %s   ·   File ▸ Open to load E57 scans.   left drag pan · right drag orbit · "
        @"right click sets orbit centre · wheel zoom · F frames all", ver::describe()];

    _spinner = [[NSProgressIndicator alloc] initWithFrame:NSZeroRect];
    _spinner.style = NSProgressIndicatorStyleSpinning;
    _spinner.hidden = YES;
    [_rightPane addSubview:_spinner];

    [self buildLayerToggles];

    // Added last so it sits above the cloud pane: the grip straddles the
    // divider, and a pane added after it would swallow half its hit area.
    _grip = [[DividerGrip alloc] initWithFrame:NSZeroRect];
    _grip.owner = self;
    [root addSubview:_grip];

    _sidebarWidth = kSidebarWidth;
    _window.contentView = root;
    [self layoutPanes];
    [_window makeKeyAndOrderFront:nil];

    NSString *err = nil;
    if (![_cloudView setupRendererReturningError:&err]) {
        NSAlert *a = [[NSAlert alloc] init];
        a.messageText = @"Could not start Metal";
        a.informativeText = err ?: @"Unknown error.";
        [a runModal];
    }
    [_window makeFirstResponder:_cloudView];
    [NSApp activateIgnoringOtherApps:YES];
}

// --- layout ---------------------------------------------------------------
//
// Everything visible is positioned here, from the window's own content bounds.
// One method, called at startup and on every resize, so the window can never be
// a size the layout has not seen.

- (void)layoutPanes {
    NSView *root = _window.contentView;
    if (!root) return;
    const CGFloat W = root.bounds.size.width;
    const CGFloat H = root.bounds.size.height;

    CGFloat side = _sidebarWidth;
    side = std::max(kSidebarMinWidth, std::min(kSidebarMaxWidth, side));
    // The cloud is never squeezed away, however narrow the window.
    side = std::min(side, std::max<CGFloat>(0, W - kDividerWidth - kCloudMinWidth));
    _sidebarWidth = side;

    _scroll.frame    = NSMakeRect(0, 0, side, H);
    _divider.frame   = NSMakeRect(side, 0, kDividerWidth, H);
    _grip.frame      = NSMakeRect(side - kGripWidth * 0.5, 0, kGripWidth, H);
    // Every pixel that is not the list or the divider is the cloud. This is the
    // line that was wrong: the right pane has to reach the window's edge, not
    // the edge of whatever size it was created at.
    _rightPane.frame = NSMakeRect(side + kDividerWidth, 0,
                                  std::max<CGFloat>(0, W - side - kDividerWidth), H);

    const CGFloat RW = _rightPane.bounds.size.width;
    _cloudView.frame = NSMakeRect(0, kStatusStripHeight, RW,
                                  std::max<CGFloat>(0, H - kStatusStripHeight));
    _status.frame    = NSMakeRect(8, 22, std::max<CGFloat>(0, RW - 50), 18);
    _progress.frame  = NSMakeRect(8, 4,  std::max<CGFloat>(0, RW - 50), 18);
    _spinner.frame   = NSMakeRect(RW - 28, 12, 18, 18);

    const CGFloat bw = 128, bh = 24, margin = 12, gap = 6;
    _cloudToggle.frame = NSMakeRect(RW - margin - bw, H - margin - bh, bw, bh);
    _voxelToggle.frame = NSMakeRect(RW - margin - bw, H - margin - 2 * bh - gap, bw, bh);
    _wrapToggle.frame  = NSMakeRect(RW - margin - bw, H - margin - 3 * bh - 2 * gap, bw, bh);

    [self layoutTableColumns];
}

// Column widths from the width actually available, so the point count cannot be
// pushed off the end of the list. Hard-coded widths were wrong twice: they have
// to fit inside the sidebar minus the scroller minus the gaps between columns,
// and that arithmetic belongs here rather than in a comment.
- (void)layoutTableColumns {
    if (_table.tableColumns.count < 3) return;
    const CGFloat scroller = _scroll.hasVerticalScroller ? 16 : 0;
    const CGFloat gaps = _table.intercellSpacing.width * 2;
    CGFloat avail = _sidebarWidth - scroller - gaps - 2;
    if (avail <= 0) return;

    NSTableColumn *name   = _table.tableColumns[0];
    NSTableColumn *status = _table.tableColumns[1];
    NSTableColumn *points = _table.tableColumns[2];

    CGFloat statusW = 104, pointsW = 62;
    if (avail < name.minWidth + statusW + pointsW) {
        // Not enough room for the preferred fixed widths: fall back to the
        // minimums, and if even those do not fit, share what there is.
        statusW = status.minWidth;
        pointsW = points.minWidth;
        if (avail < name.minWidth + statusW + pointsW) {
            const CGFloat scale = avail / (name.minWidth + statusW + pointsW);
            statusW *= scale;
            pointsW *= scale;
        }
    }
    points.width = pointsW;
    status.width = statusW;
    name.width   = std::max(name.minWidth, avail - statusW - pointsW);
}

- (void)windowDidResize:(NSNotification *)note { (void)note; [self layoutPanes]; }

// The divider is draggable: the list keeps what it is dragged to and the cloud
// takes the rest.
- (void)dragSidebarTo:(CGFloat)x {
    _sidebarWidth = x;
    [self layoutPanes];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    (void)sender; return YES;
}

// --- layer toggles --------------------------------------------------------
//
// Two push-on/push-off buttons floating over the top right of the cloud. They
// are here rather than in the menu because comparing the voxels against the
// geometry means flicking between them repeatedly, and a menu round trip for
// that gets old within a minute.

- (void)buildLayerToggles {
    struct { NSString *title; SEL action; NSButton * __strong *slot; } defs[] = {
        {@"Original clouds", @selector(toggleClouds:), &_cloudToggle},
        {@"Voxels",          @selector(toggleVoxels:), &_voxelToggle},
        {@"Shrinkwrap",      @selector(toggleWrap:),   &_wrapToggle},
    };
    for (auto &d : defs) {
        NSButton *b = [[NSButton alloc] initWithFrame:NSZeroRect];
        b.title       = d.title;
        [b setButtonType:NSButtonTypePushOnPushOff];
        // The shrinkwrap starts off, and starts unavailable. Off because it is a
        // diagnostic and a surface around the answer — left on, it would hide
        // the thing it explains. Unavailable because there is no run yet, and a
        // run on the box domain never produces one: see refreshWrapAvailability.
        const BOOL isWrap = (d.slot == &_wrapToggle);
        b.state       = isWrap ? NSControlStateValueOff : NSControlStateValueOn;
        b.enabled     = !isWrap;
        b.font        = [NSFont systemFontOfSize:11];
        b.target      = self;
        b.action      = d.action;
        // The chip is drawn here rather than by the system bezel.
        //
        // A stock push-on/push-off button shows its state by filling when on and
        // showing a plain bezel when off — and the plain bezel in dark mode is a
        // grey a shade or two off this view's background, which is a fixed
        // (0.09, 0.10, 0.12) whatever the system appearance is. Switching a layer
        // off therefore made its own button disappear, which is precisely the
        // moment you need it: the thing you have just hidden is the thing you
        // want to bring back.
        //
        // So both states are painted explicitly, against a background that is
        // known and does not follow the system: filled when on, a translucent
        // chip with a light rim when off, and legible either way.
        b.bordered    = NO;
        // MTKView forces the window layer-backed, so a sibling drawn over it
        // needs its own layer or it renders underneath.
        b.wantsLayer  = YES;
        b.layer.cornerRadius = 6;
        b.layer.borderWidth  = 1;
        [_rightPane addSubview:b];
        *d.slot = b;
        [self styleLayerToggle:b];
    }
}

// The three states of a layer toggle, painted for a dark viewport.
//
// Disabled is one of the three, and it has to be painted too: these buttons draw
// their own chip, so the system's usual greying-out of a disabled control never
// happens, and a layer with nothing in it would otherwise look exactly like one
// that is merely switched off — an invitation to keep clicking.
- (void)styleLayerToggle:(NSButton *)b {
    const BOOL on = (b.state == NSControlStateValueOn);
    // Colours in the view's own space rather than semantic ones: the background
    // they sit on is fixed, so a palette that follows the system appearance
    // would drift away from it in one direction or the other.
    NSColor *fill   = on ? [NSColor colorWithSRGBRed:0.22 green:0.47 blue:0.86 alpha:0.92]
                         : [NSColor colorWithSRGBRed:1.00 green:1.00 blue:1.00 alpha:0.10];
    NSColor *border = on ? [NSColor colorWithSRGBRed:1.00 green:1.00 blue:1.00 alpha:0.55]
                         : [NSColor colorWithSRGBRed:1.00 green:1.00 blue:1.00 alpha:0.38];
    NSColor *text   = on ? [NSColor colorWithSRGBRed:1.00 green:1.00 blue:1.00 alpha:1.00]
                         : [NSColor colorWithSRGBRed:1.00 green:1.00 blue:1.00 alpha:0.82];
    if (!b.enabled) {
        // Still readable — it says which layer is unavailable — but plainly not
        // a control at the moment: no fill, a rim dim enough to read as an
        // outline, and the label at the edge of legibility rather than past it.
        fill   = [NSColor colorWithSRGBRed:1.00 green:1.00 blue:1.00 alpha:0.03];
        border = [NSColor colorWithSRGBRed:1.00 green:1.00 blue:1.00 alpha:0.14];
        text   = [NSColor colorWithSRGBRed:1.00 green:1.00 blue:1.00 alpha:0.34];
    }
    b.layer.backgroundColor = fill.CGColor;
    b.layer.borderColor     = border.CGColor;
    // An attributed title, because a borderless button's plain `title` is drawn
    // in the system's label colour and would go dark along with the appearance.
    NSMutableParagraphStyle *para = [[NSMutableParagraphStyle alloc] init];
    para.alignment = NSTextAlignmentCenter;
    b.attributedTitle = [[NSAttributedString alloc]
        initWithString:b.title
            attributes:@{NSForegroundColorAttributeName: text,
                         NSFontAttributeName: [NSFont systemFontOfSize:11],
                         NSParagraphStyleAttributeName: para}];
}

- (void)toggleClouds:(id)sender {
    NSButton *b = (NSButton *)sender;
    _cloudView.showClouds = (b.state == NSControlStateValueOn);
    [self styleLayerToggle:b];
}

- (void)toggleVoxels:(id)sender {
    NSButton *b = (NSButton *)sender;
    _cloudView.showVoxels = (b.state == NSControlStateValueOn);
    [self styleLayerToggle:b];
}

- (void)toggleWrap:(id)sender {
    NSButton *b = (NSButton *)sender;
    _cloudView.showWrap = (b.state == NSControlStateValueOn);
    [self styleLayerToggle:b];
}

// Whether there is a wrap to show, reflected in both controls that show it.
// Turned off as well as disabled when there is none: the switch should not be
// left sitting in the on position over an empty layer.
- (void)refreshWrapAvailability {
    const BOOL have = [_cloudView hasWrap];
    _wrapToggle.enabled = have;
    if (!have) {
        _wrapToggle.state = NSControlStateValueOff;
        _cloudView.showWrap = NO;
    }
    [self styleLayerToggle:_wrapToggle];
}

- (void)buildMenu {
    _shadingItems = [NSMutableArray array];
    NSMenu *bar = [[NSMenu alloc] init];

    NSMenuItem *appItem = [[NSMenuItem alloc] init];
    NSMenu *appMenu = [[NSMenu alloc] init];
    [appMenu addItemWithTitle:@"About E57 Coverage Checker"
                       action:@selector(showAbout:) keyEquivalent:@""];
    [appMenu addItem:[NSMenuItem separatorItem]];
    [appMenu addItemWithTitle:@"Quit" action:@selector(terminate:) keyEquivalent:@"q"];
    appItem.submenu = appMenu;
    [bar addItem:appItem];

    NSMenuItem *fileItem = [[NSMenuItem alloc] init];
    NSMenu *fileMenu = [[NSMenu alloc] initWithTitle:@"File"];
    [fileMenu addItemWithTitle:@"Open…" action:@selector(openDocument:) keyEquivalent:@"o"];
    [fileMenu addItemWithTitle:@"Open Folder…" action:@selector(openFolder:) keyEquivalent:@"O"];
    [fileMenu addItem:[NSMenuItem separatorItem]];
    [fileMenu addItemWithTitle:@"Cancel" action:@selector(cancelIndexing:) keyEquivalent:@"."];
    [fileMenu addItemWithTitle:@"Close All" action:@selector(closeAll:) keyEquivalent:@"w"];
    [fileMenu addItem:[NSMenuItem separatorItem]];
    // The way out of a cache that is wrong. An index is keyed by the corpus and
    // reused on sight, so anything that leaves a store behind without finishing it
    // — a cancelled import, a crash, a full disk — locks that corpus out: every
    // later open finds the file, believes it, and never rebuilds.
    [fileMenu addItemWithTitle:@"Reindex These Scans"
                        action:@selector(reindexCurrent:) keyEquivalent:@"R"];
    [fileMenu addItemWithTitle:@"Clear Cached Indexes…"
                        action:@selector(clearCaches:) keyEquivalent:@""];
    fileItem.submenu = fileMenu;
    [bar addItem:fileItem];

    NSMenuItem *procItem = [[NSMenuItem alloc] init];
    NSMenu *procMenu = [[NSMenu alloc] initWithTitle:@"Processing"];
    [procMenu addItemWithTitle:@"Run Visibility Filter…"
                        action:@selector(runVisibilityFilter:) keyEquivalent:@"r"];
    [procMenu addItemWithTitle:@"Scan Report…" action:@selector(scanReport:) keyEquivalent:@"i"];
    [procMenu addItemWithTitle:@"Evidence Self-Test…" action:@selector(selfTest:) keyEquivalent:@"t"];
    [procMenu addItem:[NSMenuItem separatorItem]];
    // On by default. The carver is a "try" — every failure it can have comes
    // back as a declined tile that the CPU then carves — so the worst a machine
    // without a usable one suffers is the speed it would have had anyway. The
    // item is left unticked and disabled when there is genuinely no device, so
    // the menu says which case this machine is in rather than offering a switch
    // that does nothing.
    NSMenuItem *gpuItem =
        [procMenu addItemWithTitle:@"Use the GPU" action:@selector(toggleGpu:)
                     keyEquivalent:@""];
    _useGpu = ([CarveGpu shared] != nil);
    gpuItem.state = _useGpu ? NSControlStateValueOn : NSControlStateValueOff;
    // Greyed out when there is no device — see validateMenuItem:, which is what
    // actually decides. Setting `enabled` here would not survive: menus
    // autoenable, and an item whose target implements its action is switched
    // back on before it is drawn.
    if (!_useGpu) gpuItem.toolTip = [CarveGpu unavailableReason];
    [procMenu addItemWithTitle:@"Verify the GPU against the CPU"
                        action:@selector(toggleVerifyGpu:) keyEquivalent:@""];
    [procMenu addItem:[NSMenuItem separatorItem]];
    [procMenu addItemWithTitle:@"Clear Voxels" action:@selector(clearVoxels:) keyEquivalent:@""];
    procItem.submenu = procMenu;
    [bar addItem:procItem];

    NSMenuItem *viewItem = [[NSMenuItem alloc] init];
    NSMenu *viewMenu = [[NSMenu alloc] initWithTitle:@"View"];
    [viewMenu addItemWithTitle:@"Frame All" action:@selector(frameAll:) keyEquivalent:@"f"];
    [viewMenu addItemWithTitle:@"Frame Voxels" action:@selector(frameVoxels:) keyEquivalent:@"F"];
    [viewMenu addItem:[NSMenuItem separatorItem]];
    [viewMenu addItemWithTitle:@"Original Clouds"
                        action:@selector(toggleCloudsMenu:) keyEquivalent:@"1"];
    [viewMenu addItemWithTitle:@"Voxels" action:@selector(toggleVoxelsMenu:) keyEquivalent:@"2"];
    [viewMenu addItemWithTitle:@"Shrinkwrap Skin"
                        action:@selector(toggleWrapMenu:) keyEquivalent:@"3"];
    [viewMenu addItem:[NSMenuItem separatorItem]];
    // Shading. Here rather than in the run sheet because it is a way of looking
    // at the answer, not a parameter of computing it: the outward normals are
    // kept with the result, so switching costs a recolour rather than a carve.
    // Radio items, since the four are one choice.
    {
        struct { NSString *title; uint8_t mode; } modes[] = {
            {@"Shading: Lit and Height Ramp", 3},
            {@"Shading: Lit",                 1},
            {@"Shading: Height Ramp",         2},
            {@"Shading: Flat",                0},
        };
        for (auto &m : modes) {
            NSMenuItem *it = [viewMenu addItemWithTitle:m.title
                                                 action:@selector(chooseShading:)
                                          keyEquivalent:@""];
            it.tag   = m.mode;
            it.state = (m.mode == _visOptions.shading) ? NSControlStateValueOn
                                                       : NSControlStateValueOff;
            [_shadingItems addObject:it];
        }
    }
    [viewMenu addItem:[NSMenuItem separatorItem]];
    [viewMenu addItemWithTitle:@"Larger Points" action:@selector(biggerPoints:) keyEquivalent:@"]"];
    [viewMenu addItemWithTitle:@"Smaller Points" action:@selector(smallerPoints:) keyEquivalent:@"["];
    viewItem.submenu = viewMenu;
    [bar addItem:viewItem];

    NSApp.mainMenu = bar;
}

// --- actions --------------------------------------------------------------

- (void)openDocument:(id)sender {
    (void)sender;
    if (_busy) return;
    NSOpenPanel *panel = [NSOpenPanel openPanel];
    panel.allowsMultipleSelection = YES;
    panel.canChooseDirectories = NO;
    UTType *e57Type = [UTType typeWithFilenameExtension:@"e57"];
    if (e57Type) panel.allowedContentTypes = @[e57Type];
    panel.message = @"Choose E57 scans.";
    if ([panel runModal] != NSModalResponseOK) return;

    NSMutableArray<NSString *> *paths = [NSMutableArray array];
    for (NSURL *u in panel.URLs) [paths addObject:u.path];
    [self loadPaths:paths];
}

// A thousand-setup job is a directory, not a multi-select.
- (void)openFolder:(id)sender {
    (void)sender;
    if (_busy) return;
    NSOpenPanel *panel = [NSOpenPanel openPanel];
    panel.canChooseFiles = NO;
    panel.canChooseDirectories = YES;
    panel.message = @"Choose a folder of E57 scans.";
    if ([panel runModal] != NSModalResponseOK) return;

    NSMutableArray<NSString *> *paths = [NSMutableArray array];
    for (NSURL *dir in panel.URLs) {
        NSDirectoryEnumerator *e = [NSFileManager.defaultManager enumeratorAtPath:dir.path];
        for (NSString *rel in e)
            if ([rel.pathExtension caseInsensitiveCompare:@"e57"] == NSOrderedSame)
                [paths addObject:[dir.path stringByAppendingPathComponent:rel]];
    }
    if (paths.count == 0) {
        _status.stringValue = @"No .e57 files in that folder.";
        return;
    }
    [self loadPaths:paths];
}

- (void)cancelIndexing:(id)sender {
    (void)sender;
    if (_cancel) _cancel->store(true);
}

- (void)closeAll:(id)sender {
    (void)sender;
    if (_busy) return;
    _survey = indexer::Survey{};
    _paths.clear();
    [_table reloadData];
    [_cloudView closeAll];
    _status.stringValue = @"Closed.";
    _progress.stringValue = @"";
}

- (void)frameAll:(id)sender { (void)sender; [_cloudView frameAll]; }
- (void)frameVoxels:(id)sender { (void)sender; [_cloudView frameVoxels]; }

// The menu items and the buttons are two faces of one switch, so each keeps the
// other in step rather than the two drifting apart.
- (void)toggleCloudsMenu:(id)sender {
    (void)sender;
    const BOOL on = !_cloudView.showClouds;
    _cloudView.showClouds = on;
    _cloudToggle.state = on ? NSControlStateValueOn : NSControlStateValueOff;
    [self styleLayerToggle:_cloudToggle];
}

- (void)toggleVoxelsMenu:(id)sender {
    (void)sender;
    const BOOL on = !_cloudView.showVoxels;
    _cloudView.showVoxels = on;
    _voxelToggle.state = on ? NSControlStateValueOn : NSControlStateValueOff;
    [self styleLayerToggle:_voxelToggle];
}

- (void)toggleWrapMenu:(id)sender {
    (void)sender;
    const BOOL on = !_cloudView.showWrap;
    _cloudView.showWrap = on;
    _wrapToggle.state = on ? NSControlStateValueOn : NSControlStateValueOff;
    [self styleLayerToggle:_wrapToggle];
}

// The scan report, in the app. Everything `e57cov info` prints — the raster,
// the recovered angular mapping and whether it describes that raster, the blind
// cone and which way up the instrument was, the decode cross-checks.
//
// Here because the CLI is not what gets run: Xcode builds the scheme you have
// selected, so building the app never builds the command line tool, and a stale
// e57cov on a path is indistinguishable from a current one.
- (void)scanReport:(id)sender {
    (void)sender;
    if (_busy) return;
    if (_paths.empty()) {
        _status.stringValue = @"Open some E57 scans first.";
        return;
    }

    _busy = YES;
    _spinner.hidden = NO;
    [_spinner startAnimation:nil];
    _status.stringValue = @"Reading the scans…";

    report::Options ro;
    ro.maxRange         = _visOptions.maxRange;
    ro.blindCone        = _visOptions.blindCone;

    auto paths = std::make_shared<std::vector<std::string>>(_paths);
    __weak AppDelegate *weakSelf = self;

    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
      @autoreleasepool {
        auto text = std::make_shared<std::string>();
        report::scanReport(*paths, ro, *text);

        dispatch_async(dispatch_get_main_queue(), ^{
            AppDelegate *me = weakSelf;
            if (!me) return;
            me->_busy = NO;
            [me->_spinner stopAnimation:nil];
            me->_spinner.hidden = YES;

            NSString *body = [NSString stringWithUTF8String:text->c_str()] ?: @"(unreadable)";

            // Written to a file as well as shown. A report you have to
            // transcribe out of a window is a report that arrives wrong.
            NSString *dir = NSSearchPathForDirectoriesInDomains(
                NSDesktopDirectory, NSUserDomainMask, YES).firstObject
                ?: NSTemporaryDirectory();
            NSString *file = [dir stringByAppendingPathComponent:@"e57cov-scan-report.txt"];
            NSError *werr = nil;
            const BOOL wrote = [body writeToFile:file atomically:YES
                                        encoding:NSUTF8StringEncoding error:&werr];

            [me showReport:body savedTo:(wrote ? file : nil)];
            me->_status.stringValue = wrote
                ? [NSString stringWithFormat:@"Scan report written to %@", file]
                : @"Scan report ready (could not write a file)";
        });
      }
    });
}

// The same window, for the self-test: the evidence primitive asked about each
// scan's own cells, where the right answer is not in doubt, plus whether the
// setups are where the files say they are.
//
// Here rather than only in the CLI for the same reason the scan report is: the
// command line tool is not what gets run, and a diagnostic nobody can reach is
// not a diagnostic.
- (void)selfTest:(id)sender {
    (void)sender;
    if (_busy) return;
    if (_paths.empty()) {
        _status.stringValue = @"Open some E57 scans first.";
        return;
    }

    _busy = YES;
    _spinner.hidden = NO;
    [_spinner startAnimation:nil];
    _status.stringValue = @"Testing the evidence path…";

    report::Options ro;
    ro.maxRange         = _visOptions.maxRange;
    ro.blindCone        = _visOptions.blindCone;

    auto paths = std::make_shared<std::vector<std::string>>(_paths);
    __weak AppDelegate *weakSelf = self;

    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
      @autoreleasepool {
        auto text = std::make_shared<std::string>();
        report::selfTest(*paths, ro, *text);

        dispatch_async(dispatch_get_main_queue(), ^{
            AppDelegate *me = weakSelf;
            if (!me) return;
            me->_busy = NO;
            [me->_spinner stopAnimation:nil];
            me->_spinner.hidden = YES;

            NSString *body = [NSString stringWithUTF8String:text->c_str()] ?: @"(unreadable)";
            NSString *dir = NSSearchPathForDirectoriesInDomains(
                NSDesktopDirectory, NSUserDomainMask, YES).firstObject
                ?: NSTemporaryDirectory();
            NSString *file = [dir stringByAppendingPathComponent:@"e57cov-self-test.txt"];
            NSError *werr = nil;
            const BOOL wrote = [body writeToFile:file atomically:YES
                                        encoding:NSUTF8StringEncoding error:&werr];
            [me showReport:body savedTo:(wrote ? file : nil)];
            me->_status.stringValue = wrote
                ? [NSString stringWithFormat:@"Self-test written to %@", file]
                : @"Self-test ready (could not write a file)";
        });
      }
    });
}

// A plain scrollable text window with a Copy button. Selectable and monospaced,
// because the whole point is getting the text somewhere else intact.
- (void)showReport:(NSString *)body savedTo:(NSString *)file {
    const NSRect r = NSMakeRect(0, 0, 900, 640);
    NSWindow *w = [[NSWindow alloc]
        initWithContentRect:r
                  styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                             NSWindowStyleMaskResizable)
                    backing:NSBackingStoreBuffered
                      defer:NO];
    w.title = file ? [NSString stringWithFormat:@"Scan Report — %@", file.lastPathComponent]
                   : @"Scan Report";
    w.releasedWhenClosed = NO;

    NSView *root = [[NSView alloc] initWithFrame:r];
    NSScrollView *sv = [[NSScrollView alloc] initWithFrame:NSMakeRect(0, 40, r.size.width,
                                                                     r.size.height - 40)];
    sv.hasVerticalScroller = YES;
    sv.hasHorizontalScroller = YES;
    sv.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;

    NSTextView *tv = [[NSTextView alloc] initWithFrame:sv.bounds];
    tv.editable = NO;
    tv.selectable = YES;
    tv.richText = NO;
    tv.font = [NSFont monospacedSystemFontOfSize:11 weight:NSFontWeightRegular];
    tv.string = body;
    tv.minSize = NSMakeSize(0, 0);
    tv.maxSize = NSMakeSize(FLT_MAX, FLT_MAX);
    tv.verticallyResizable = YES;
    tv.horizontallyResizable = YES;
    tv.textContainer.widthTracksTextView = NO;
    tv.textContainer.containerSize = NSMakeSize(FLT_MAX, FLT_MAX);
    sv.documentView = tv;
    [root addSubview:sv];

    NSButton *copy = [[NSButton alloc] initWithFrame:NSMakeRect(r.size.width - 120, 8, 108, 24)];
    copy.title = @"Copy all";
    copy.bezelStyle = NSBezelStyleRounded;
    copy.target = self;
    copy.action = @selector(copyReport:);
    copy.autoresizingMask = NSViewMinXMargin;
    [root addSubview:copy];

    w.contentView = root;
    [w center];
    [w makeKeyAndOrderFront:nil];
    _reportWindow = w;
    _reportText = tv;
}

- (void)copyReport:(id)sender {
    (void)sender;
    if (!_reportText) return;
    NSPasteboard *pb = NSPasteboard.generalPasteboard;
    [pb clearContents];
    [pb setString:_reportText.string forType:NSPasteboardTypeString];
    _status.stringValue = @"Scan report copied to the clipboard.";
}

- (void)showAbout:(id)sender {
    (void)sender;
    [NSApp orderFrontStandardAboutPanel:@{
        NSAboutPanelOptionApplicationVersion: [NSString stringWithUTF8String:ver::describe()],
    }];
}

// Throws away this corpus's cached index and builds it again.
//
// The targeted escape hatch, and the one to reach for first: a cancelled import
// leaves a store that the corpus key then finds and trusts, so the scans that were
// interrupted are the exact scans that can never be re-opened. This deletes that
// one store and re-runs the open that would otherwise have been skipped.
- (void)reindexCurrent:(id)sender {
    (void)sender;
    if (_busy) { _status.stringValue = @"Busy — cancel first."; return; }
    if (_paths.empty()) { _status.stringValue = @"No scans open to reindex."; return; }

    NSString *storePath = [self storePathForKey:corpusKey(_paths)];
    NSError *rmErr = nil;
    const BOOL existed = [NSFileManager.defaultManager fileExistsAtPath:storePath];
    if (existed && ![NSFileManager.defaultManager removeItemAtPath:storePath error:&rmErr]) {
        _status.stringValue = [NSString stringWithFormat:@"Could not remove the cached index: %@",
                               rmErr.localizedDescription];
        return;
    }
    // Chunk spill files live beside the store and are named after it. A cancelled
    // build leaves them, and they are megabytes each.
    [self removeSpillFilesBeside:storePath];

    NSMutableArray<NSString *> *again = [NSMutableArray array];
    for (const std::string &p : _paths) [again addObject:ns(p)];
    _status.stringValue = existed ? @"Cached index discarded — rebuilding."
                                  : @"No cached index for these scans — indexing.";
    [self loadPaths:again];
}

// Everything cached, for every corpus. The blunt instrument, for when it is not
// clear which index is the bad one.
- (void)clearCaches:(id)sender {
    (void)sender;
    if (_busy) { _status.stringValue = @"Busy — cancel first."; return; }

    NSString *dir = [self cacheDirectory];
    NSFileManager *fm = NSFileManager.defaultManager;
    NSArray<NSString *> *names = [fm contentsOfDirectoryAtPath:dir error:nil];

    uint64_t bytes = 0;
    NSUInteger files = 0;
    for (NSString *name in names) {
        NSDictionary *at = [fm attributesOfItemAtPath:[dir stringByAppendingPathComponent:name]
                                                error:nil];
        if (at) { bytes += at.fileSize; ++files; }
    }
    if (files == 0) { _status.stringValue = @"Nothing cached."; return; }

    NSAlert *a = [[NSAlert alloc] init];
    a.messageText = @"Clear cached indexes?";
    a.informativeText = [NSString stringWithFormat:
        @"Deletes %lu cached file(s), %.1f MB, from\n%@\n\n"
         "Nothing is lost that cannot be rebuilt from the E57 files — an index is a "
         "cache of them. The next open of any corpus will rebuild it, which takes as "
         "long as the first open did.\n\n"
         "Do this when an import is refusing to re-run: a store left behind by a "
         "cancelled or crashed build is found by the corpus key and trusted, and the "
         "scans it covers cannot be opened again until it is gone.",
        (unsigned long)files, double(bytes) / (1024.0 * 1024.0), dir];
    [a addButtonWithTitle:@"Clear"];
    [a addButtonWithTitle:@"Cancel"];
    if ([a runModal] != NSAlertFirstButtonReturn) return;

    NSUInteger removed = 0, failed = 0;
    for (NSString *name in names) {
        NSError *e = nil;
        if ([fm removeItemAtPath:[dir stringByAppendingPathComponent:name] error:&e]) ++removed;
        else ++failed;
    }
    // The survey's per-file verdicts go with it. They are keyed by path, size and
    // mtime so they were never the stale thing, but "clear the cache" that leaves
    // a cache behind is a worse answer than a slower next open.
    _status.stringValue = failed
        ? [NSString stringWithFormat:@"Cleared %lu cached file(s); %lu could not be removed.",
           (unsigned long)removed, (unsigned long)failed]
        : [NSString stringWithFormat:@"Cleared %lu cached file(s), %.1f MB. The next open "
                                      "rebuilds.", (unsigned long)removed,
           double(bytes) / (1024.0 * 1024.0)];
}

// Chunk spill files are named after the store they were built for and live beside
// it. A finished build removes its own; a cancelled one cannot.
- (void)removeSpillFilesBeside:(NSString *)storePath {
    NSString *dir  = [storePath stringByDeletingLastPathComponent];
    NSString *base = [storePath lastPathComponent];
    NSFileManager *fm = NSFileManager.defaultManager;
    for (NSString *name in [fm contentsOfDirectoryAtPath:dir error:nil]) {
        if (![name hasPrefix:base] || [name isEqualToString:base]) continue;
        [fm removeItemAtPath:[dir stringByAppendingPathComponent:name] error:nil];
    }
}

- (void)clearVoxels:(id)sender {
    (void)sender;
    [_cloudView clearVoxels];
    [self refreshWrapAvailability];
    _status.stringValue = @"Voxels cleared.";
}

// A switch with nothing to switch to should not look available. Two items can be
// in that position — the GPU carve with no device, and the wrap on a run that
// used the box domain — so everything else validates through.
- (BOOL)validateMenuItem:(NSMenuItem *)item {
    if (item.action == @selector(toggleGpu:)) return [CarveGpu shared] != nil;
    if (item.action == @selector(toggleWrapMenu:)) return [_cloudView hasWrap];
    // Both cache actions rewrite files an in-flight index is using, and reindexing
    // needs something open to reindex.
    if (item.action == @selector(reindexCurrent:)) return !_busy && !_paths.empty();
    if (item.action == @selector(clearCaches:))    return !_busy;
    return YES;
}

- (void)toggleGpu:(id)sender {
    _useGpu = !_useGpu;
    ((NSMenuItem *)sender).state = _useGpu ? NSControlStateValueOn : NSControlStateValueOff;
    if (_useGpu && ![CarveGpu shared]) {
        _useGpu = NO;
        ((NSMenuItem *)sender).state = NSControlStateValueOff;
        _status.stringValue = [NSString stringWithFormat:@"No GPU carve available: %@",
                               [CarveGpu unavailableReason]];
        return;
    }
    _status.stringValue = _useGpu ? @"Visibility filter will run on the GPU."
                                  : @"Visibility filter will run on the CPU.";
}

// One of four, so the four keep each other in step.
- (void)chooseShading:(id)sender {
    NSMenuItem *picked = (NSMenuItem *)sender;
    _visOptions.shading = uint8_t(picked.tag);
    for (NSMenuItem *it in _shadingItems)
        it.state = (it == picked) ? NSControlStateValueOn : NSControlStateValueOff;
    // Applies to what is already on screen. A carve is minutes and a recolour is
    // a pass over the drawn voxels, so this is not a setting that waits for the
    // next run to mean anything.
    [_cloudView setVoxelShading:_visOptions.shading];
}

- (void)toggleVerifyGpu:(id)sender {
    _verifyGpu = !_verifyGpu;
    ((NSMenuItem *)sender).state = _verifyGpu ? NSControlStateValueOn : NSControlStateValueOff;
    _status.stringValue = _verifyGpu
        ? @"Every tile will be carved twice and the results compared. Half speed."
        : @"GPU verification off.";
}
- (void)biggerPoints:(id)sender { (void)sender; _cloudView.pointSize = _cloudView.pointSize + 0.5f; }
- (void)smallerPoints:(id)sender { (void)sender; _cloudView.pointSize = _cloudView.pointSize - 0.5f; }

- (void)application:(NSApplication *)sender openFiles:(NSArray<NSString *> *)filenames {
    [self loadPaths:filenames];
    [sender replyToOpenOrPrint:NSApplicationDelegateReplySuccess];
}

// --- the two-phase open ---------------------------------------------------

- (NSString *)cacheDirectory {
    NSArray *dirs = NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES);
    NSString *base = dirs.count ? dirs[0] : NSTemporaryDirectory();
    NSString *dir = [base stringByAppendingPathComponent:@"E57CoverageChecker"];
    [NSFileManager.defaultManager createDirectoryAtPath:dir withIntermediateDirectories:YES
                                             attributes:nil error:nil];
    return dir;
}

- (NSString *)storePathForKey:(const std::string &)key {
    return [[self cacheDirectory] stringByAppendingPathComponent:ns(key + ".lod")];
}

// Where per-file scan verdicts are remembered. ONE file for all corpora, not one
// per corpus, and that is the whole point: the store is keyed over the corpus so
// adding a scan invalidates it, while a verdict about a file keeps as long as the
// file does. Entries are keyed by path, size and mtime — see
// indexer::SurveyOptions::cachePath.
- (NSString *)checkCachePath {
    return [[self cacheDirectory] stringByAppendingPathComponent:@"scan-checks.txt"];
}

- (void)loadPaths:(NSArray<NSString *> *)paths {
    if (_busy || paths.count == 0) return;

    std::vector<std::string> cpaths;
    for (NSString *p in paths) {
        const char *u = p.UTF8String;      // nil for an undecodable path
        if (u) cpaths.push_back(u);
    }
    if (cpaths.empty()) { _status.stringValue = @"No readable paths."; return; }
    _paths = cpaths;

    _busy = YES;
    _cancel = std::make_shared<std::atomic<bool>>(false);
    _spinner.hidden = NO;
    [_spinner startAnimation:nil];
    _status.stringValue = [NSString stringWithFormat:@"Reading headers from %zu file%s…",
                           cpaths.size(), cpaths.size() == 1 ? "" : "s"];

    // Everything the worker needs is captured by value up front. Nothing on the
    // background queue touches `self` or an ivar: reaching through `self->` from
    // another thread is how a window closing mid-index turns into a crash, and
    // it was being done in twenty-odd places.
    NSString *storePath = [self storePathForKey:corpusKey(cpaths)];
    const BOOL cached = [NSFileManager.defaultManager fileExistsAtPath:storePath];
    auto cancel = _cancel;

    __weak AppDelegate *weakSelf = self;

    // One place where progress crosses back to the main thread, and it
    // re-checks that the delegate is still alive every time.
    void (^report)(NSString *) = ^(NSString *line) {
        dispatch_async(dispatch_get_main_queue(), ^{
            AppDelegate *me = weakSelf;
            if (me) me->_progress.stringValue = line;
        });
    };
    void (^finish)(NSString *) = ^(NSString *line) {
        dispatch_async(dispatch_get_main_queue(), ^{
            AppDelegate *me = weakSelf;
            if (!me) return;
            me->_status.stringValue = line;
            me->_progress.stringValue = @"";
            me->_busy = NO;
            [me->_spinner stopAnimation:nil];
            me->_spinner.hidden = YES;
        });
    };

    // Captures by value only — no reference into block storage, and no ObjC
    // state reached through a dangling `self`.
    auto makeProgress = [report, cancel](NSString *prefix, uint64_t every) {
        return [report, cancel, prefix, every](const std::string &stage,
                                               uint64_t done, uint64_t total) {
            if (every == 0 || done % every == 0) {
                @autoreleasepool {
                    NSString *what = prefix ?: [NSString stringWithUTF8String:stage.c_str()];
                    report([NSString stringWithFormat:@"%@  %llu / %llu", what,
                            (unsigned long long)done, (unsigned long long)total]);
                }
            }
            return !cancel->load();
        };
    };

    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
      @autoreleasepool {
        // --- stage 1: headers only ---
        indexer::SurveyOptions so;      // classify off: no point decoding
        indexer::Survey survey = indexer::survey(cpaths, so, makeProgress(nil, 25));
        if (cancel->load()) { finish(@"Cancelled."); return; }

        std::vector<double> setups;
        setups.reserve(survey.scans.size() * 3);
        for (const auto &sc : survey.scans)
            if (sc.usable) for (int k = 0; k < 3; ++k) setups.push_back(sc.setup[k]);

        const size_t usable = survey.usableCount();
        const size_t files  = survey.filesRead;
        const std::string declared = humanCount(survey.totalPoints());

        dispatch_async(dispatch_get_main_queue(), ^{
            AppDelegate *me = weakSelf;
            if (!me) return;
            me->_survey = survey;
            [me->_table reloadData];
            [me->_cloudView setSetups:setups];
            me->_status.stringValue = [NSString stringWithFormat:
                @"%zu setups from %zu files   ·   %s declared points   ·   indexing…",
                usable, files, declared.c_str()];
        });

        // --- stage 2: the point store ---
        if (!cached) {
            // The build needs the full structured check, which decodes a sample
            // per scan; the fast survey deliberately skipped it.
            //
            // Threads left at the default, which is the hardware's count: files
            // are checked in parallel and the answer does not depend on how many
            // at once. See indexer::SurveyOptions::threads.
            //
            // And the verdicts are remembered per file, so adding a scan to a
            // large corpus re-checks the scan rather than the corpus. The store
            // below still has to be rebuilt — an octree over the corpus is a
            // corpus-wide thing — but that is the unavoidable half.
            indexer::SurveyOptions full;
            full.classify  = true;
            full.cachePath = [self checkCachePath].UTF8String;
            indexer::Survey checked =
                indexer::survey(cpaths, full, makeProgress(@"checking scans", 10));
            if (cancel->load()) { finish(@"Cancelled."); return; }
            const size_t reused = checked.filesFromCache;
            const size_t fresh  = checked.filesRead > reused ? checked.filesRead - reused : 0;

            dispatch_async(dispatch_get_main_queue(), ^{
                AppDelegate *me = weakSelf;
                if (!me) return;
                me->_survey = checked;
                [me->_table reloadData];
                // What the check actually cost, which is otherwise invisible:
                // on an incremental corpus nearly every file is reused, and a
                // figure that drops to zero unexpectedly is the visible symptom
                // of something rewriting the files.
                if (reused)
                    me->_progress.stringValue = [NSString stringWithFormat:
                        @"checked %zu scan file(s), reused %zu from cache", fresh, reused];
            });

            indexer::BuildOptions bo;
            indexer::BuildStats stats;
            std::string err;
            const bool ok = indexer::build(checked, storePath.UTF8String, bo, stats,
                                           makeProgress(nil, 0), err);
            if (!ok) {
                // A partial store must not be left where the cache key would
                // find it and present it as complete.
                [NSFileManager.defaultManager removeItemAtPath:storePath error:nil];
                finish(cancel->load() ? @"Cancelled."
                                      : [NSString stringWithFormat:@"Indexing failed: %s",
                                         err.c_str()]);
                return;
            }
            if (stats.outsideRoot || stats.dropped) {
                report([NSString stringWithFormat:
                    @"indexed with losses: %llu outside bounds, %llu dropped at depth limit",
                    (unsigned long long)stats.outsideRoot, (unsigned long long)stats.dropped]);
            }
        }

        dispatch_async(dispatch_get_main_queue(), ^{
            AppDelegate *me = weakSelf;
            if (!me) return;
            NSString *openErr = nil;
            if (![me->_cloudView openStore:storePath error:&openErr]) {
                // A cached index that will not open is a dead end, and the corpus
                // key will find it again on every future attempt — so the scans
                // that failed once can never be opened again. Discard it and say
                // what to do, rather than reporting the error and leaving the
                // thing that caused it in place.
                //
                // Only when it came FROM the cache. A store this run just built
                // and cannot open is a real fault worth seeing, and deleting the
                // evidence would turn it into an infinite rebuild.
                if (cached) {
                    [NSFileManager.defaultManager removeItemAtPath:storePath error:nil];
                    [me removeSpillFilesBeside:storePath];
                    me->_status.stringValue = [NSString stringWithFormat:
                        @"The cached index was unusable (%@) and has been discarded — "
                         "File ▸ Reindex These Scans, or just open them again.", openErr];
                } else {
                    me->_status.stringValue =
                        [NSString stringWithFormat:@"Could not open store: %@", openErr];
                }
            } else {
                me->_status.stringValue = [NSString stringWithFormat:
                    @"%zu setups   ·   store %@   ·   drag to navigate",
                    usable, cached ? @"reused from cache" : @"built"];
            }
            me->_progress.stringValue = @"";
            me->_busy = NO;
            [me->_spinner stopAnimation:nil];
            me->_spinner.hidden = YES;
        });
      }
    });
}

// --- the visibility filter ------------------------------------------------
//
// Same pipeline as `e57cov carve`: src/visibility.{h,cpp} does the work, and
// this is the window's way of asking for it. It runs on a background queue for
// the same reason indexing does — a full site at 5 cm is minutes of arithmetic
// and a beachball is not a progress report.
//
// The parameters are asked for rather than assumed. Voxel size and range change
// both the answer and how long it takes by orders of magnitude, and a filter
// that silently picked them would be reporting on a question the operator never
// asked.

- (NSTextField *)labelWithText:(NSString *)text frame:(NSRect)frame {
    NSTextField *f = [[NSTextField alloc] initWithFrame:frame];
    f.stringValue = text;
    f.bezeled = NO; f.editable = NO; f.drawsBackground = NO;
    f.font = [NSFont systemFontOfSize:11];
    f.textColor = [NSColor secondaryLabelColor];
    return f;
}

- (NSTextField *)fieldWithValue:(NSString *)value frame:(NSRect)frame {
    NSTextField *f = [[NSTextField alloc] initWithFrame:frame];
    f.stringValue = value;
    f.font = [NSFont monospacedDigitSystemFontOfSize:12 weight:NSFontWeightRegular];
    f.alignment = NSTextAlignmentRight;
    return f;
}

- (void)runVisibilityFilter:(id)sender {
    (void)sender;
    if (_busy) return;
    if (_paths.empty()) {
        _status.stringValue = @"Open some E57 scans first.";
        return;
    }

    // --- parameters -------------------------------------------------------
    // Laid out from the top down, in one place, because it is a fixed stack and
    // adding a control by eyeballing a y is how the region popup ended up drawn
    // over the fourth parameter row.
    //
    //   270 242 214 186 158 130   six label/value rows, 28 apart
    //    99                        the region popup, 24 tall, clearing 130 by seven
    //    74  52  30   8            four tick boxes, 22 apart
    NSView *acc = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 460, 298)];
    struct { NSString *label; NSString *value; } rows[] = {
        {@"Voxel size (m)",     [NSString stringWithFormat:@"%.3f", _visOptions.voxelSize]},
        {@"Maximum range (m)",  [NSString stringWithFormat:@"%.1f", _visOptions.maxRange]},
        {@"Tile size (voxels)", [NSString stringWithFormat:@"%u", _visOptions.tileVoxels]},
        // The shrinkwrap's buffer and the box's margin are the same number meaning
        // the same thing — how far past the last return the question still applies
        // — so the label names both rather than making it look like two settings.
        // Negative is allowed and useful: it pulls the question inside the walls.
        {@"Buffer / margin past the last return (m, may be −)",
                                [NSString stringWithFormat:@"%.1f", _visOptions.domainMargin]},
        // The cap on voxels DRAWN, not on voxels found. Over it the frontier is
        // sampled, which is the one setting whose effect looks like a bug: a
        // large site at 5 cm has tens of millions of frontier voxels and six
        // million of them scattered through a building reads as nothing at all.
        {@"Voxels drawn at most (millions)",
                                [NSString stringWithFormat:@"%.0f",
                                 double(_visOptions.displayCap) / 1048576.0]},
        // The instrument's rated MINIMUM range. A surface closer than this returns
        // nothing, and believing that empty cell clears a pencil of space straight
        // through the surface — a setup parked half a metre from a wall otherwise
        // carves a fan out through it to the maximum range. Here rather than
        // buried because it is a property of the instrument the operator knows and
        // the tool cannot read from the file.
        {@"Instrument minimum range (m, 0 to ignore)",
                                [NSString stringWithFormat:@"%.2f", _visOptions.minRange]},
    };
    NSMutableArray<NSTextField *> *fields = [NSMutableArray array];
    for (int i = 0; i < 6; ++i) {
        const CGFloat y = 270 - i * 28;
        [acc addSubview:[self labelWithText:rows[i].label frame:NSMakeRect(0, y, 220, 20)]];
        NSTextField *f = [self fieldWithValue:rows[i].value frame:NSMakeRect(230, y - 3, 90, 22)];
        [acc addSubview:f];
        [fields addObject:f];
    }

    // What region the question covers — the setting that changes the answer more
    // than any other, so it is a choice rather than a tick box.
    [acc addSubview:[self labelWithText:@"Region" frame:NSMakeRect(0, 102, 60, 20)]];
    NSPopUpButton *region =
        [[NSPopUpButton alloc] initWithFrame:NSMakeRect(62, 99, 396, 24) pullsDown:NO];
    [region addItemsWithTitles:@[@"Shrinkwrap of the returns (tightest)",
                                 @"Box around the surveyed extent",
                                 @"Everything in range of a setup"]];
    const vis::DomainMode regionOrder[3] = {vis::DomainMode::Shrinkwrap,
                                            vis::DomainMode::MeasuredExtent,
                                            vis::DomainMode::RangeSpheres};
    for (int i = 0; i < 3; ++i)
        if (regionOrder[i] == _visOptions.domain) [region selectItemAtIndex:i];
    [acc addSubview:region];

    NSButton *interior = [[NSButton alloc] initWithFrame:NSMakeRect(0, 74, 460, 20)];
    interior.title = @"Scanned entirely indoors (leave the space outside the walls out) "
                     @"— or give the margin above a minus sign";
    [interior setButtonType:NSButtonTypeSwitch];
    interior.font = [NSFont systemFontOfSize:11];
    interior.state = _visOptions.wrapInteriorOnly ? NSControlStateValueOn
                                                  : NSControlStateValueOff;
    [acc addSubview:interior];

    NSButton *firstHit = [[NSButton alloc] initWithFrame:NSMakeRect(0, 52, 460, 20)];
    firstHit.title = @"Stop at the first evidence (faster; visible and occupied become "
                     @"lower bounds)";
    [firstHit setButtonType:NSButtonTypeSwitch];
    firstHit.font = [NSFont systemFontOfSize:11];
    firstHit.state = (_visOptions.earlyOut == carve::EarlyOut::AnyEvidence)
                   ? NSControlStateValueOn : NSControlStateValueOff;
    [acc addSubview:firstHit];

    NSButton *solid = [[NSButton alloc] initWithFrame:NSMakeRect(0, 30, 460, 20)];
    solid.title = @"Show every unobserved voxel, not just the frontier";
    [solid setButtonType:NSButtonTypeSwitch];
    solid.font = [NSFont systemFontOfSize:11];
    solid.state = _visOptions.solid ? NSControlStateValueOn : NSControlStateValueOff;
    [acc addSubview:solid];

    // The intersection. A display filter over a finished carve rather than a
    // parameter of it, but it belongs here: it only means anything when the region
    // is the shrinkwrap, and it is the difference between seeing the site and
    // seeing a solid red mass in front of it.
    NSButton *insideWrap = [[NSButton alloc] initWithFrame:NSMakeRect(0, 8, 460, 20)];
    insideWrap.title = @"Keep only unobserved voxels inside the shrinkwrap "
                       @"(drops the blanket running out to the range limit)";
    [insideWrap setButtonType:NSButtonTypeSwitch];
    insideWrap.font = [NSFont systemFontOfSize:11];
    insideWrap.state = _intersectWrap ? NSControlStateValueOn : NSControlStateValueOff;
    [acc addSubview:insideWrap];
    // Shading is not here. It is in the View menu, because it changes how the
    // answer is drawn rather than what the answer is, and it applies to a
    // finished carve without running another one.

    NSAlert *a = [[NSAlert alloc] init];
    a.messageText = @"Run visibility filter";
    a.informativeText =
        @"Marks every voxel some setup could see through or measured a surface in, and "
        @"reports the rest: space in range of a scanner that nothing observed.\n\n"
        @"Limiting to the surveyed extent asks only about a box around what the scans "
        @"actually returned. Leave it on for an interior job: the range spheres otherwise "
        @"reach tens of metres out through every wall, and the answer becomes mostly sky.\n\n"
        @"The instrument minimum range matters more than it looks: a surface closer than "
        @"that returns nothing, and an empty cell that is really a wall at arm's length "
        @"would otherwise clear space straight through it, out to the maximum range.\n\n"
        @"By default only the frontier of that space is drawn — where coverage stops. "
        @"The full volume hides its own interior anyway, and there is far more of it.\n\n"
        @"This is the CPU reference, so a large site at 5 cm takes minutes. "
        @"File ▸ Cancel stops it, and a coarser voxel is much faster: halving the "
        @"voxel size costs eight times the work.";
    [a addButtonWithTitle:@"Run"];
    [a addButtonWithTitle:@"Cancel"];
    a.accessoryView = acc;
    if ([a runModal] != NSAlertFirstButtonReturn) return;

    vis::Options opt = _visOptions;
    const double voxel = fields[0].doubleValue;
    const double range = fields[1].doubleValue;
    const long   tile   = fields[2].integerValue;
    const double margin = fields[3].doubleValue;
    const double drawnM = fields[4].doubleValue;
    const double minRng = fields[5].doubleValue;
    // The margin may be NEGATIVE — see vis::Options::domainMargin. On an indoor
    // job that is how the space past the walls is left out of the question in the
    // first place, rather than filtered out of the answer afterwards.
    if (!(voxel > 0.0) || !(range > 0.0) || tile <= 0 || tile > 4096 ||
        !std::isfinite(margin) || !(minRng >= 0.0) || minRng >= range) {
        _status.stringValue =
            @"Voxel size and range must be positive, tile size 1–4096, and the "
             "instrument minimum range between zero and the maximum. The margin "
             "may be negative, which pulls the question inside the walls.";
        return;
    }
    opt.voxelSize    = voxel;
    opt.maxRange     = range;
    opt.tileVoxels   = uint32_t(tile);
    opt.domainMargin = margin;
    opt.minRange     = minRng;
    if (drawnM > 0.0)
        opt.displayCap = uint64_t(std::min(drawnM, 512.0) * 1048576.0);
    {
        const vis::DomainMode order[3] = {vis::DomainMode::Shrinkwrap,
                                          vis::DomainMode::MeasuredExtent,
                                          vis::DomainMode::RangeSpheres};
        const NSInteger i = region.indexOfSelectedItem;
        opt.domain = (i >= 0 && i < 3) ? order[i] : vis::DomainMode::Shrinkwrap;
    }
    opt.wrapInteriorOnly = (interior.state == NSControlStateValueOn);
    _intersectWrap   = (insideWrap.state == NSControlStateValueOn);
    const BOOL intersectWrap = _intersectWrap;
    opt.solid        = (solid.state == NSControlStateValueOn);
    opt.earlyOut     = (firstHit.state == NSControlStateValueOn)
                     ? carve::EarlyOut::AnyEvidence : carve::EarlyOut::Saturated;
    // The carver is a "try": every failure it can have comes back as a declined
    // tile and that tile is carved on the CPU, so switching this on can slow the
    // run down but cannot change the answer.
    CarveGpu *gpu = _useGpu ? [CarveGpu shared] : nil;
    opt.carver       = gpu ? [CarveGpu carver] : nullptr;
    opt.carverUser   = gpu ? (__bridge void *)gpu : nullptr;
    opt.verifyCarver = gpu && _verifyGpu;
    _visOptions      = opt;

    // --- run --------------------------------------------------------------
    // Everything the worker needs is captured by value; nothing on the
    // background queue reaches through `self`.
    _busy = YES;
    _cancel = std::make_shared<std::atomic<bool>>(false);
    _spinner.hidden = NO;
    [_spinner startAnimation:nil];
    _status.stringValue = @"Running the visibility filter…";

    auto cancel = _cancel;
    auto paths  = std::make_shared<std::vector<std::string>>(_paths);
    __weak AppDelegate *weakSelf = self;

    void (^report)(NSString *) = ^(NSString *line) {
        dispatch_async(dispatch_get_main_queue(), ^{
            AppDelegate *me = weakSelf;
            if (me) me->_progress.stringValue = line;
        });
    };
    void (^finish)(NSString *) = ^(NSString *line) {
        dispatch_async(dispatch_get_main_queue(), ^{
            AppDelegate *me = weakSelf;
            if (!me) return;
            me->_status.stringValue = line;
            me->_progress.stringValue = @"";
            me->_busy = NO;
            [me->_spinner stopAnimation:nil];
            me->_spinner.hidden = YES;
        });
    };

    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
      @autoreleasepool {
        auto result = std::make_shared<vis::Result>();
        std::string err;

        // Progress crosses to the main thread at most every 200 ms: the carve
        // ticks per tile, and a dispatch per tile would cost more than the
        // arithmetic it is reporting on.
        auto lastPost = std::chrono::steady_clock::now();
        vis::Progress progress = [report, cancel, &lastPost]
                                 (const std::string &stage, uint64_t done, uint64_t total) {
            const auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - lastPost).count() > 0.2 || done == total) {
                lastPost = now;
                @autoreleasepool {
                    report([NSString stringWithFormat:@"%s  %llu / %llu",
                            stage.c_str(), (unsigned long long)done,
                            (unsigned long long)total]);
                }
            }
            return !cancel->load();
        };

        const bool ok = vis::run(*paths, opt, progress, *result, err);
        if (!ok) {
            finish(cancel->load() ? @"Visibility filter cancelled."
                                  : [NSString stringWithFormat:@"Visibility filter failed: %s",
                                     err.c_str()]);
            return;
        }

        // The intersection, applied to the finished carve. Here rather than inside
        // vis::run because it changes what is DRAWN and not what was measured: the
        // volume and the percentage on the status line below stay the figures for
        // the whole unobserved set, which is the honest number, while the display
        // drops the part of it running out to the range limit.
        uint64_t voxelsBeforeWrapFilter = result->voxels.size();
        if (intersectWrap && !result->wrapGrid.empty())
            vis::keepVoxelsInsideWrap(*result);
        const uint64_t voxelsDrawn = result->voxels.size();

        const double vol = result->unknownVolume();
        const double pct = result->stats.reachable
                         ? 100.0 * double(result->stats.unknown) / double(result->stats.reachable)
                         : 0.0;
        // The options struct is copied into the block, so `opt` here still
        // carries the carver that was installed above; the result knows how many
        // tiles it actually took.
        NSString *note = result->note.empty() ? @""
                       : [NSString stringWithFormat:@"   ·   %s", result->note.c_str()];
        // A setup whose mapping was refused is in the list and contributes
        // nothing; one mounted upside down is worth knowing about. Neither is
        // visible from the totals, and both change how the picture reads.
        NSString *warn = @"";
        if (result->setupsWithoutMapping) {
            warn = [warn stringByAppendingFormat:
                    @"   ·   ⚠︎ %llu setup(s) contribute NOTHING (mapping refused)",
                    (unsigned long long)result->setupsWithoutMapping];
            // With the reason. A run that produces no voxels at all and says only
            // "refused" sends you off to run something else to find out why.
            if (!result->mappingRefusedWhy.empty())
                warn = [warn stringByAppendingFormat:@" — %s",
                        result->mappingRefusedWhy.c_str()];
        }
        if (result->setupsInverted)
            warn = [warn stringByAppendingFormat:@"   ·   %llu inverted",
                    (unsigned long long)result->setupsInverted];
        // A corpus too large for the image budget gets its rasters coarsened, and
        // coarse cells clear less space — so the unobserved volume on this line is
        // overstated by however much. That is a different answer, not a blurrier
        // one, and it has to be on the line the number is on.
        // What the wrap was built from, and that it can be looked at.
        //
        // The occupancy count is here because it is the one number that catches
        // the wrap being built from less than the survey measured — which is a
        // mistake this has already made, by sampling the raster at a stride.
        // Under-marking shrinks the domain, and a smaller domain quietly removes
        // questions instead of answering them, so it never looks like an error on
        // its own. Against the cell size and the site it is a number a reader can
        // sanity-check; against the skin on screen it is a number they can see.
        if (result->domain.kind == carve::Domain::Kind::Wrap && !result->wrapGrid.empty())
            warn = [warn stringByAppendingFormat:
                    @"   ·   wrap: %llu cells at %.2f m, from %llu holding returns (⌘3)",
                    (unsigned long long)result->wrapGrid.domainCells, result->wrapGrid.cell,
                    (unsigned long long)result->wrapGrid.occupiedCells];
        if (result->domain.kind == carve::Domain::Kind::Wrap && result->wrapGrid.interiorOnly &&
            !result->wrapGrid.sealLeaked)
            warn = [warn stringByAppendingFormat:
                    @"   ·   interior only: %llu wrap cells outside the shell dropped",
                    (unsigned long long)result->wrapGrid.droppedOutside];
        // Decimation, said out loud. The CLI has always reported it and the app
        // never did, so a run whose frontier is larger than the display cap drew
        // one voxel in however many and looked like a run that found nothing —
        // which at 5 cm over a large building is the normal case, not an edge one.
        if (result->keptFraction < 0.999 && result->qualified)
            warn = [warn stringByAppendingFormat:
                    @"   ·   ⚠︎ DRAWING 1 VOXEL IN %.0f — %llu of %llu unobserved voxels, "
                     "sampled to fit the display cap of %.1f M. Raise it, or use a coarser "
                     "voxel; the volume and percentage above are the whole set and are right",
                    result->keptFraction > 0 ? 1.0 / result->keptFraction : 0.0,
                    (unsigned long long)result->voxels.size(),
                    (unsigned long long)result->qualified,
                    double(opt.displayCap) / 1048576.0];
        if (intersectWrap && voxelsBeforeWrapFilter != voxelsDrawn)
            warn = [warn stringByAppendingFormat:
                    @"   ·   drawing %llu of %llu unobserved voxels — those inside the "
                     "shrinkwrap; the volume above is still the whole set",
                    (unsigned long long)voxelsDrawn,
                    (unsigned long long)voxelsBeforeWrapFilter];
        else if (intersectWrap && result->wrapGrid.empty())
            warn = [warn stringByAppendingString:
                    @"   ·   ⚠︎ no shrinkwrap to intersect with — pick the shrinkwrap region"];
        if (!result->wrapNote.empty())
            warn = [warn stringByAppendingFormat:@"   ·   ⚠︎ %s", result->wrapNote.c_str()];
        if (result->setupsBinned)
            warn = [warn stringByAppendingFormat:
                    @"   ·   ⚠︎ %llu raster(s) COARSENED up to %ux to fit memory — "
                     "unobserved volume is overstated; raise the image budget",
                    (unsigned long long)result->setupsBinned, result->worstBinStep];
        // A band that is there and was left believed clears a cone straight at its
        // pole, because the elevation table is extrapolated across the band and a
        // direction near the pole maps into it. At the HIGH end that is the sky
        // which carves the volume above a site, and is what should happen; at the
        // LOW end it is a cone through whatever the instrument was standing on, so
        // that one is a warning and the other is a count.
        // Setups parked inside their own minimum range of something. Every cell of
        // such a zone would otherwise have cleared to the rated range through the
        // surface that was too close to measure, which is the largest single way
        // this answer can be wrong in the optimistic direction.
        if (result->setupsTooClose)
            warn = [warn stringByAppendingFormat:
                    @"   ·   %llu setup(s) parked inside the %.2f m minimum range of "
                     "something: %llu cells demoted, nearest border %.2f m — each would "
                     "have cleared to %.0f m through the surface in the way",
                    (unsigned long long)result->setupsTooClose, opt.minRange,
                    (unsigned long long)result->tooCloseCells, result->tooCloseNearest,
                    opt.maxRange];
        if (result->setupsBandBelievedLow)
            warn = [warn stringByAppendingFormat:
                    @"   ·   ⚠︎ UNSAMPLED BAND BELIEVED AS SKY at the LOW end of %llu setup(s) "
                     "— each clears a cone straight down, through a floor",
                    (unsigned long long)result->setupsBandBelievedLow];
        if (result->setupsBandBelievedHigh)
            warn = [warn stringByAppendingFormat:
                    @"   ·   sky believed at the high end of %llu setup(s)",
                    (unsigned long long)result->setupsBandBelievedHigh];
        // Where each scan put its own cone. Every scan decides from its own raster,
        // so this is a tally rather than one verdict, and a scan that identified
        // nothing is the one worth flagging: every empty cell in it clears.
        warn = [warn stringByAppendingFormat:
                @"   ·   cone: %llu at the start, %llu at the end, %llu at both ends, "
                 "%llu by angle",
                (unsigned long long)result->coneVerdict.atFirst,
                (unsigned long long)result->coneVerdict.atLast,
                (unsigned long long)result->coneVerdict.bothEnds,
                (unsigned long long)result->coneVerdict.byAngle];
        if (result->coneVerdict.undecided)
            warn = [warn stringByAppendingFormat:
                    @"   ·   ⚠︎ no cone identified on %llu setup(s) — every empty cell in "
                     "those clears to the rated range",
                    (unsigned long long)result->coneVerdict.undecided];

        // Which path actually ran, and — when asked to check it — what the check
        // found. A verification that prints nothing is indistinguishable from one
        // that never ran, so it says so either way, pass or fail. Same for a
        // carver that declined every tile: the run is still correct, because the
        // CPU caught them, but "CPU" alone would hide that the GPU was asked and
        // said no.
        NSString *carver = @"CPU";
        if (opt.carver) {
            if (result->carverTiles == 0)
                carver = [NSString stringWithFormat:
                          @"⚠︎ the GPU declined all %llu tiles — the CPU carved them",
                          (unsigned long long)result->carverRefused];
            else if (result->carverRefused)
                carver = [NSString stringWithFormat:@"%llu tiles on the GPU, %llu on the CPU",
                          (unsigned long long)result->carverTiles,
                          (unsigned long long)result->carverRefused];
            else
                carver = [NSString stringWithFormat:@"%llu tiles on the GPU",
                          (unsigned long long)result->carverTiles];
            if (opt.verifyCarver) {
                const double frac = result->carverVoxelsCompared
                                  ? double(result->carverDisagreements) /
                                    double(result->carverVoxelsCompared)
                                  : 0.0;
                // What a disagreement rate means, which a bare "FAILED" does not
                // say. The kernel works in float and the CPU in double, and the
                // raster cell a voxel lands in is chosen by a bin index of tens
                // of thousands — which float resolves to about a
                // four-hundredth of a bin. A voxel that close to a cell edge can
                // land either side of it, and then reads a neighbouring cell
                // holding a different distance.
                //
                // That is not a fault and cannot be fixed in float: measured by
                // replaying both paths over a real scan, it puts 0.014% of voxels
                // in a different cell, and every one of them sits within 0.006 of
                // a bin of an edge. A kernel that is actually wrong — a bad
                // index, a bad transform — misses by whole cells and shows up
                // percent-wide. So the two are separated here rather than both
                // being called failure.
                if (result->carverVoxelsCompared == 0)
                    carver = [carver stringByAppendingString:
                              @", verified against the CPU: ⚠︎ NOTHING WAS COMPARED"];
                else if (result->carverDisagreements == 0)
                    carver = [carver stringByAppendingFormat:
                              @", verified: %.1f M voxels against the CPU, every one identical",
                              double(result->carverVoxelsCompared) / 1e6];
                else if (frac <= kCarverFloatNoise)
                    carver = [carver stringByAppendingFormat:
                              @", verified: %llu of %.1f M voxels differ (%.4f%%, %.2f m³) — "
                               "float32 rounding at raster cell edges, not a fault",
                              (unsigned long long)result->carverDisagreements,
                              double(result->carverVoxelsCompared) / 1e6, 100.0 * frac,
                              double(result->carverDisagreements) * opt.voxelSize *
                                  opt.voxelSize * opt.voxelSize];
                else
                    carver = [carver stringByAppendingFormat:
                              @", ⚠︎ VERIFICATION FAILED: %llu of %.1f M voxels differ (%.4f%%) "
                               "— far more than float32 rounding explains",
                              (unsigned long long)result->carverDisagreements,
                              double(result->carverVoxelsCompared) / 1e6, 100.0 * frac];
            }
        }

        NSString *line = [NSString stringWithFormat:
            @"%llu setups   ·   %.0f m³ unobserved (%.1f%% of what was in range)   ·   "
            @"%zu voxels drawn   ·   %@%@%@%@",
            (unsigned long long)result->setupsUsed, vol, pct, result->voxels.size(),
            carver,
            result->partial ? @"   ·   PARTIAL RUN" : @"", warn, note];

        dispatch_async(dispatch_get_main_queue(), ^{
            AppDelegate *me = weakSelf;
            if (!me) return;
            [me->_cloudView setVoxelResult:*result];
            me->_voxelToggle.state = NSControlStateValueOn;
            [me styleLayerToggle:me->_voxelToggle];
            me->_cloudView.showVoxels = YES;
            // A run on the box domain builds no wrap, so there is nothing for
            // the switch to show. Disabled rather than left to click and do
            // nothing visible, which reads as a broken wrap.
            [me refreshWrapAvailability];
            me->_status.stringValue = line;
            me->_progress.stringValue = @"";
            me->_busy = NO;
            [me->_spinner stopAnimation:nil];
            me->_spinner.hidden = YES;
        });
      }
    });
}

// --- viewer callback ------------------------------------------------------

- (void)cloudViewDidChangeView:(NSString *)status {
    if (!_busy) _status.stringValue = status;
}

// --- table ----------------------------------------------------------------

- (NSInteger)numberOfRowsInTableView:(NSTableView *)tableView {
    (void)tableView;
    return (NSInteger)_survey.scans.size();
}

- (NSView *)tableView:(NSTableView *)tableView
   viewForTableColumn:(NSTableColumn *)column
                  row:(NSInteger)row {
    (void)tableView;
    if (row < 0 || (size_t)row >= _survey.scans.size()) return nil;
    const indexer::ScanRef &e = _survey.scans[(size_t)row];

    NSTextField *label = [[NSTextField alloc] initWithFrame:NSMakeRect(0, 0, column.width, 26)];
    label.bezeled = NO; label.editable = NO; label.drawsBackground = NO;
    label.lineBreakMode = NSLineBreakByTruncatingMiddle;
    label.font = [NSFont systemFontOfSize:11];

    if ([column.identifier isEqualToString:@"scan"]) {
        label.stringValue = ns(e.name);
        label.toolTip = ns(e.path);
    } else if ([column.identifier isEqualToString:@"status"]) {
        label.stringValue = ns(e.status.empty() ? kindLabel(e.kind) : e.status);
        label.toolTip = ns(e.status);
        switch (e.kind) {
        case check::Kind::Structured: label.textColor = [NSColor systemGreenColor];  break;
        case check::Kind::Unified:    label.textColor = [NSColor systemRedColor];    break;
        case check::Kind::Ambiguous:  label.textColor = [NSColor systemOrangeColor]; break;
        }
    } else {
        label.stringValue = ns(humanCount(e.recordCount));
        label.alignment = NSTextAlignmentRight;
        label.textColor = [NSColor secondaryLabelColor];
    }
    return label;
}

@end


@implementation DividerGrip

- (void)resetCursorRects {
    [self addCursorRect:self.bounds cursor:NSCursor.resizeLeftRightCursor];
}

// Dragging reports a position in the container's coordinates, which is exactly
// what the sidebar width is measured in — no accumulated deltas to drift.
- (void)mouseDragged:(NSEvent *)event {
    const NSPoint p = [self.superview convertPoint:event.locationInWindow fromView:nil];
    [self.owner dragSidebarTo:p.x];
}

@end

// --- entry point ----------------------------------------------------------

int main(int argc, const char *argv[]) {
    (void)argc; (void)argv;
    @autoreleasepool {
        NSApplication *app = [NSApplication sharedApplication];
        [app setActivationPolicy:NSApplicationActivationPolicyRegular];
        // NSApplication.delegate is a WEAK property, and a scope-local strong
        // reference whose last use is this assignment can be released by ARC
        // immediately — leaving NSApp.delegate dangling before the run loop
        // even starts. Held for the process lifetime instead.
        static AppDelegate *delegate = nil;
        delegate = [[AppDelegate alloc] init];
        app.delegate = delegate;
        [app run];
    }
    return 0;
}
