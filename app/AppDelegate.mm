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

#import "CloudView.h"

#include "../src/indexer.h"
#include "../src/point_store.h"
#include "../src/scan_check.h"

#include <algorithm>
#include <atomic>
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

const char *kindLabel(check::Kind k) {
    switch (k) {
    case check::Kind::Structured: return "structured";
    case check::Kind::Unified:    return "merged — not indexed";
    case check::Kind::Ambiguous:  return "ambiguous";
    }
    return "?";
}

} // namespace

@interface AppDelegate : NSObject <NSApplicationDelegate, NSTableViewDataSource,
                                   NSTableViewDelegate, CloudViewDelegate>
@end

@implementation AppDelegate {
    NSWindow            *_window;
    CloudView           *_cloudView;
    NSTableView         *_table;
    NSTextField         *_status;
    NSTextField         *_progress;
    NSProgressIndicator *_spinner;

    indexer::Survey      _survey;
    BOOL                 _busy;
    // Shared with the worker rather than read through `self`: it is written on
    // the main thread and read on a background queue, which as a plain BOOL was
    // a data race.
    std::shared_ptr<std::atomic<bool>> _cancel;
}

- (void)applicationDidFinishLaunching:(NSNotification *)note {
    (void)note;
    [self buildMenu];

    const NSRect frame = NSMakeRect(0, 0, 1400, 900);
    _window = [[NSWindow alloc]
        initWithContentRect:frame
                  styleMask:(NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                             NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable)
                    backing:NSBackingStoreBuffered
                      defer:NO];
    _window.title = @"E57 Coverage Checker";
    [_window center];

    NSSplitView *split = [[NSSplitView alloc] initWithFrame:frame];
    split.vertical = YES;
    split.dividerStyle = NSSplitViewDividerStyleThin;
    split.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;

    _table = [[NSTableView alloc] initWithFrame:NSZeroRect];
    _table.dataSource = self;
    _table.delegate = self;
    _table.usesAlternatingRowBackgroundColors = YES;
    _table.rowHeight = 30;
    struct { NSString *ident; NSString *title; CGFloat width; } cols[] = {
        {@"scan",   @"Setup",  230},
        {@"status", @"Status", 170},
        {@"points", @"Points", 90},
    };
    for (auto &c : cols) {
        NSTableColumn *col = [[NSTableColumn alloc] initWithIdentifier:c.ident];
        col.title = c.title;
        col.width = c.width;
        [_table addTableColumn:col];
    }
    NSScrollView *scroll = [[NSScrollView alloc] initWithFrame:NSMakeRect(0, 0, 520, 900)];
    scroll.documentView = _table;
    scroll.hasVerticalScroller = YES;
    scroll.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;

    NSView *rightPane = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 880, 900)];
    rightPane.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;

    _cloudView = [[CloudView alloc] initWithFrame:NSMakeRect(0, 44, 880, 856)];
    _cloudView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    _cloudView.cloudDelegate = self;
    [rightPane addSubview:_cloudView];

    _status = [[NSTextField alloc] initWithFrame:NSMakeRect(8, 22, 830, 18)];
    _progress = [[NSTextField alloc] initWithFrame:NSMakeRect(8, 4, 830, 18)];
    for (NSTextField *f in @[_status, _progress]) {
        f.bezeled = NO; f.editable = NO; f.drawsBackground = NO;
        f.font = [NSFont monospacedDigitSystemFontOfSize:11 weight:NSFontWeightRegular];
        f.textColor = [NSColor secondaryLabelColor];
        f.autoresizingMask = NSViewWidthSizable | NSViewMaxYMargin;
        [rightPane addSubview:f];
    }
    _status.stringValue = @"File ▸ Open to load E57 scans.   left drag pan · right drag orbit · "
                          @"right click sets orbit centre · wheel zoom · F frames all";

    _spinner = [[NSProgressIndicator alloc] initWithFrame:NSMakeRect(852, 12, 18, 18)];
    _spinner.style = NSProgressIndicatorStyleSpinning;
    _spinner.hidden = YES;
    _spinner.autoresizingMask = NSViewMinXMargin | NSViewMaxYMargin;
    [rightPane addSubview:_spinner];

    [split addSubview:scroll];
    [split addSubview:rightPane];
    [split setPosition:520 ofDividerAtIndex:0];

    _window.contentView = split;
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

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    (void)sender; return YES;
}

- (void)buildMenu {
    NSMenu *bar = [[NSMenu alloc] init];

    NSMenuItem *appItem = [[NSMenuItem alloc] init];
    NSMenu *appMenu = [[NSMenu alloc] init];
    [appMenu addItemWithTitle:@"About E57 Coverage Checker"
                       action:@selector(orderFrontStandardAboutPanel:) keyEquivalent:@""];
    [appMenu addItem:[NSMenuItem separatorItem]];
    [appMenu addItemWithTitle:@"Quit" action:@selector(terminate:) keyEquivalent:@"q"];
    appItem.submenu = appMenu;
    [bar addItem:appItem];

    NSMenuItem *fileItem = [[NSMenuItem alloc] init];
    NSMenu *fileMenu = [[NSMenu alloc] initWithTitle:@"File"];
    [fileMenu addItemWithTitle:@"Open…" action:@selector(openDocument:) keyEquivalent:@"o"];
    [fileMenu addItemWithTitle:@"Open Folder…" action:@selector(openFolder:) keyEquivalent:@"O"];
    [fileMenu addItem:[NSMenuItem separatorItem]];
    [fileMenu addItemWithTitle:@"Cancel Indexing" action:@selector(cancelIndexing:) keyEquivalent:@"."];
    [fileMenu addItemWithTitle:@"Close All" action:@selector(closeAll:) keyEquivalent:@"w"];
    fileItem.submenu = fileMenu;
    [bar addItem:fileItem];

    NSMenuItem *viewItem = [[NSMenuItem alloc] init];
    NSMenu *viewMenu = [[NSMenu alloc] initWithTitle:@"View"];
    [viewMenu addItemWithTitle:@"Frame All" action:@selector(frameAll:) keyEquivalent:@"f"];
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
    [_table reloadData];
    [_cloudView closeAll];
    _status.stringValue = @"Closed.";
    _progress.stringValue = @"";
}

- (void)frameAll:(id)sender { (void)sender; [_cloudView frameAll]; }
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
