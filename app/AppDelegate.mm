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
    case check::Kind::Unified:    return "merged — not indexed";
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
    };
    for (auto &d : defs) {
        NSButton *b = [[NSButton alloc] initWithFrame:NSZeroRect];
        b.title       = d.title;
        b.bezelStyle  = NSBezelStyleRounded;
        [b setButtonType:NSButtonTypePushOnPushOff];
        b.state       = NSControlStateValueOn;
        b.font        = [NSFont systemFontOfSize:11];
        b.target      = self;
        b.action      = d.action;
        // MTKView forces the window layer-backed, so a sibling drawn over it
        // needs its own layer or it renders underneath.
        b.wantsLayer  = YES;
        [_rightPane addSubview:b];
        *d.slot = b;
    }
}

- (void)toggleClouds:(id)sender {
    _cloudView.showClouds = (((NSButton *)sender).state == NSControlStateValueOn);
}

- (void)toggleVoxels:(id)sender {
    _cloudView.showVoxels = (((NSButton *)sender).state == NSControlStateValueOn);
}

- (void)buildMenu {
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
    fileItem.submenu = fileMenu;
    [bar addItem:fileItem];

    NSMenuItem *procItem = [[NSMenuItem alloc] init];
    NSMenu *procMenu = [[NSMenu alloc] initWithTitle:@"Processing"];
    [procMenu addItemWithTitle:@"Run Visibility Filter…"
                        action:@selector(runVisibilityFilter:) keyEquivalent:@"r"];
    [procMenu addItemWithTitle:@"Scan Report…" action:@selector(scanReport:) keyEquivalent:@"i"];
    [procMenu addItemWithTitle:@"Evidence Self-Test…" action:@selector(selfTest:) keyEquivalent:@"t"];
    [procMenu addItem:[NSMenuItem separatorItem]];
    [procMenu addItemWithTitle:@"Use the GPU" action:@selector(toggleGpu:) keyEquivalent:@""];
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
}

- (void)toggleVoxelsMenu:(id)sender {
    (void)sender;
    const BOOL on = !_cloudView.showVoxels;
    _cloudView.showVoxels = on;
    _voxelToggle.state = on ? NSControlStateValueOn : NSControlStateValueOff;
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
    ro.noReturnRadius   = _visOptions.skyRadius;
    ro.noReturnFraction = _visOptions.skyFraction;

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
    ro.noReturnRadius   = _visOptions.skyRadius;
    ro.noReturnFraction = _visOptions.skyFraction;

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

- (void)clearVoxels:(id)sender {
    (void)sender;
    [_cloudView clearVoxels];
    _status.stringValue = @"Voxels cleared.";
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

- (NSString *)storePathForKey:(const std::string &)key {
    NSArray *dirs = NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES);
    NSString *base = dirs.count ? dirs[0] : NSTemporaryDirectory();
    NSString *dir = [base stringByAppendingPathComponent:@"E57CoverageChecker"];
    [NSFileManager.defaultManager createDirectoryAtPath:dir withIntermediateDirectories:YES
                                             attributes:nil error:nil];
    return [dir stringByAppendingPathComponent:ns(key + ".lod")];
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
            indexer::SurveyOptions full;
            full.classify = true;
            indexer::Survey checked =
                indexer::survey(cpaths, full, makeProgress(@"checking scans", 10));
            if (cancel->load()) { finish(@"Cancelled."); return; }

            dispatch_async(dispatch_get_main_queue(), ^{
                AppDelegate *me = weakSelf;
                if (!me) return;
                me->_survey = checked;
                [me->_table reloadData];
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
                me->_status.stringValue =
                    [NSString stringWithFormat:@"Could not open store: %@", openErr];
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
    NSView *acc = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 460, 212)];
    struct { NSString *label; NSString *value; } rows[] = {
        {@"Voxel size (m)",     [NSString stringWithFormat:@"%.3f", _visOptions.voxelSize]},
        {@"Maximum range (m)",  [NSString stringWithFormat:@"%.1f", _visOptions.maxRange]},
        {@"Tile size (voxels)", [NSString stringWithFormat:@"%u", _visOptions.tileVoxels]},
        {@"Margin past the last return (m)",
                                [NSString stringWithFormat:@"%.1f", _visOptions.domainMargin]},
    };
    NSMutableArray<NSTextField *> *fields = [NSMutableArray array];
    for (int i = 0; i < 4; ++i) {
        const CGFloat y = 184 - i * 28;
        [acc addSubview:[self labelWithText:rows[i].label frame:NSMakeRect(0, y, 220, 20)]];
        NSTextField *f = [self fieldWithValue:rows[i].value frame:NSMakeRect(230, y - 3, 90, 22)];
        [acc addSubview:f];
        [fields addObject:f];
    }

    NSButton *extent = [[NSButton alloc] initWithFrame:NSMakeRect(0, 74, 460, 20)];
    extent.title = @"Limit to the surveyed extent (recommended for interiors)";
    [extent setButtonType:NSButtonTypeSwitch];
    extent.font = [NSFont systemFontOfSize:11];
    extent.state = (_visOptions.domain == vis::DomainMode::MeasuredExtent)
                 ? NSControlStateValueOn : NSControlStateValueOff;
    [acc addSubview:extent];

    NSButton *firstHit = [[NSButton alloc] initWithFrame:NSMakeRect(0, 52, 400, 20)];
    firstHit.title = @"Stop at the first evidence (faster; visible and occupied become "
                     @"lower bounds)";
    [firstHit setButtonType:NSButtonTypeSwitch];
    firstHit.font = [NSFont systemFontOfSize:11];
    firstHit.state = (_visOptions.earlyOut == carve::EarlyOut::AnyEvidence)
                   ? NSControlStateValueOn : NSControlStateValueOff;
    [acc addSubview:firstHit];

    NSButton *solid = [[NSButton alloc] initWithFrame:NSMakeRect(0, 8, 400, 20)];
    solid.title = @"Show every unobserved voxel, not just the frontier";
    [solid setButtonType:NSButtonTypeSwitch];
    solid.font = [NSFont systemFontOfSize:11];
    solid.state = _visOptions.solid ? NSControlStateValueOn : NSControlStateValueOff;
    [acc addSubview:solid];

    NSAlert *a = [[NSAlert alloc] init];
    a.messageText = @"Run visibility filter";
    a.informativeText =
        @"Marks every voxel some setup could see through or measured a surface in, and "
        @"reports the rest: space in range of a scanner that nothing observed.\n\n"
        @"Limiting to the surveyed extent asks only about a box around what the scans "
        @"actually returned. Leave it on for an interior job: the range spheres otherwise "
        @"reach tens of metres out through every wall, and the answer becomes mostly sky.\n\n"
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
    if (!(voxel > 0.0) || !(range > 0.0) || tile <= 0 || tile > 4096 || margin < 0.0) {
        _status.stringValue =
            @"Voxel size and range must be positive, tile size 1–4096, margin not negative.";
        return;
    }
    opt.voxelSize    = voxel;
    opt.maxRange     = range;
    opt.tileVoxels   = uint32_t(tile);
    opt.domainMargin = margin;
    opt.domain       = (extent.state == NSControlStateValueOn)
                     ? vis::DomainMode::MeasuredExtent : vis::DomainMode::RangeSpheres;
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
        if (result->setupsBinned)
            warn = [warn stringByAppendingFormat:
                    @"   ·   ⚠︎ %llu raster(s) COARSENED up to %ux to fit memory — "
                     "unobserved volume is overstated; raise the image budget",
                    (unsigned long long)result->setupsBinned, result->worstBinStep];
        // Which end of the raster the blind cone is at decides whether a band of
        // empty cells clears space to the rated range or establishes nothing, and
        // nothing else on this line changes the picture as much. Undecided leaves
        // both ends believed, which carves a cone through the ground under every
        // setup — a warning, not a footnote.
        if (!result->coneVerdict.decided)
            warn = [warn stringByAppendingString:
                    @"   ·   ⚠︎ blind cone NOT identified — empty cells at both ends "
                     "of the raster are believed"];
        else
            warn = [warn stringByAppendingFormat:@"   ·   cone at the %@ of the raster",
                    result->coneVerdict.atFirstRow ? @"start" : @"end"];

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
                if (result->carverVoxelsCompared == 0)
                    carver = [carver stringByAppendingString:
                              @", verified against the CPU: ⚠︎ NOTHING WAS COMPARED"];
                else if (result->carverDisagreements == 0)
                    carver = [carver stringByAppendingFormat:
                              @", verified: %.1f M voxels against the CPU, every one identical",
                              double(result->carverVoxelsCompared) / 1e6];
                else
                    carver = [carver stringByAppendingFormat:
                              @", ⚠︎ VERIFICATION FAILED: %llu of %.1f M voxels differ (%.4f%%)",
                              (unsigned long long)result->carverDisagreements,
                              double(result->carverVoxelsCompared) / 1e6,
                              100.0 * double(result->carverDisagreements) /
                                      double(result->carverVoxelsCompared)];
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
            me->_cloudView.showVoxels = YES;
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
