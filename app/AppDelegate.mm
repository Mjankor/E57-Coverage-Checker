// Window, file list, and loading.
//
// Loading is on a background queue: a structured scan is tens of millions of
// points and decoding several files would otherwise beachball the app for
// minutes with no indication of progress.
//
// Scans that fail the structured check are listed with their reason but are
// not added to the scene. The visibility method needs a single known origin per
// scan; a merged cloud silently produces nonsense rather than failing, so the
// place to stop it is here, on load.

#import <Cocoa/Cocoa.h>
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>

#import "CloudView.h"

#include "../src/e57.h"
#include "../src/picker.h"
#include "../src/point_cloud.h"
#include "../src/scan_check.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

struct Entry {
    std::string file;        // display name only
    std::string path;
    std::string scan;
    check::Kind kind = check::Kind::Ambiguous;
    std::string status;
    std::string detail;
    uint64_t    sourcePoints = 0;
    size_t      loadedPoints = 0;
    bool        rendered = false;
};

NSString *ns(const std::string &s) { return [NSString stringWithUTF8String:s.c_str()]; }

std::string humanCount(uint64_t n) {
    char buf[64];
    if (n >= 1000000ull) std::snprintf(buf, sizeof(buf), "%.1f M", double(n) / 1e6);
    else if (n >= 1000ull) std::snprintf(buf, sizeof(buf), "%.1f k", double(n) / 1e3);
    else std::snprintf(buf, sizeof(buf), "%llu", (unsigned long long)n);
    return buf;
}

} // namespace

@interface AppDelegate : NSObject <NSApplicationDelegate, NSTableViewDataSource,
                                   NSTableViewDelegate, CloudViewDelegate>
@end

@implementation AppDelegate {
    NSWindow        *_window;
    CloudView       *_cloudView;
    NSTableView     *_table;
    NSTextField     *_status;
    NSProgressIndicator *_spinner;
    std::vector<Entry>              _entries;
    std::vector<viewer::PointCloud> _clouds;
    BOOL                            _loading;
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

    // --- left: scan list ---
    _table = [[NSTableView alloc] initWithFrame:NSZeroRect];
    _table.dataSource = self;
    _table.delegate = self;
    _table.usesAlternatingRowBackgroundColors = YES;
    _table.rowHeight = 34;

    struct { NSString *ident; NSString *title; CGFloat width; } cols[] = {
        {@"scan",   @"Scan",   210},
        {@"status", @"Status", 190},
        {@"points", @"Points", 110},
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

    // --- right: viewer + status bar ---
    NSView *rightPane = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 880, 900)];
    rightPane.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;

    _cloudView = [[CloudView alloc] initWithFrame:NSMakeRect(0, 28, 880, 872)];
    _cloudView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    _cloudView.cloudDelegate = self;
    [rightPane addSubview:_cloudView];

    _status = [[NSTextField alloc] initWithFrame:NSMakeRect(8, 4, 700, 20)];
    _status.bezeled = NO;
    _status.editable = NO;
    _status.drawsBackground = NO;
    _status.font = [NSFont monospacedDigitSystemFontOfSize:11 weight:NSFontWeightRegular];
    _status.textColor = [NSColor secondaryLabelColor];
    _status.autoresizingMask = NSViewWidthSizable | NSViewMaxYMargin;
    _status.stringValue = @"File ▸ Open to load structured E57 scans.   "
                          @"left drag pan · right drag orbit · right click sets orbit centre · wheel zoom · F frames all";
    [rightPane addSubview:_status];

    _spinner = [[NSProgressIndicator alloc] initWithFrame:NSMakeRect(0, 0, 18, 18)];
    _spinner.style = NSProgressIndicatorStyleSpinning;
    _spinner.hidden = YES;
    _spinner.autoresizingMask = NSViewMinXMargin | NSViewMaxYMargin;
    _spinner.frame = NSMakeRect(852, 5, 18, 18);
    [rightPane addSubview:_spinner];

    [split addSubview:scroll];
    [split addSubview:rightPane];
    [split setPosition:520 ofDividerAtIndex:0];

    _window.contentView = split;
    [_window makeKeyAndOrderFront:nil];

    NSString *err = nil;
    if (![_cloudView setupRendererReturningError:&err]) {
        // Without Metal there is nothing to show, and pretending otherwise
        // leaves an empty black window with no explanation.
        NSAlert *a = [[NSAlert alloc] init];
        a.messageText = @"Could not start Metal";
        a.informativeText = err ?: @"Unknown error.";
        [a runModal];
    }
    [_window makeFirstResponder:_cloudView];
    [NSApp activateIgnoringOtherApps:YES];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    (void)sender;
    return YES;
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
    if (_loading) return;
    NSOpenPanel *panel = [NSOpenPanel openPanel];
    panel.allowsMultipleSelection = YES;
    panel.canChooseDirectories = NO;
    UTType *e57Type = [UTType typeWithFilenameExtension:@"e57"];
    if (e57Type) panel.allowedContentTypes = @[e57Type];
    panel.message = @"Choose one or more structured E57 scans.";
    if ([panel runModal] != NSModalResponseOK) return;

    NSMutableArray<NSString *> *paths = [NSMutableArray array];
    for (NSURL *u in panel.URLs) [paths addObject:u.path];
    [self loadPaths:paths];
}

- (void)closeAll:(id)sender {
    (void)sender;
    if (_loading) return;
    _entries.clear();
    _clouds.clear();
    [_table reloadData];
    [_cloudView setScene:std::vector<viewer::PointCloud>{}];
    _status.stringValue = @"Closed all scans.";
}

- (void)frameAll:(id)sender { (void)sender; [_cloudView frameAll]; }
- (void)biggerPoints:(id)sender { (void)sender; _cloudView.pointSize = _cloudView.pointSize + 0.5f; }
- (void)smallerPoints:(id)sender { (void)sender; _cloudView.pointSize = _cloudView.pointSize - 0.5f; }

- (BOOL)application:(NSApplication *)sender openFile:(NSString *)filename {
    (void)sender;
    [self loadPaths:@[filename]];
    return YES;
}

- (void)application:(NSApplication *)sender openFiles:(NSArray<NSString *> *)filenames {
    (void)sender;
    [self loadPaths:filenames];
    [sender replyToOpenOrPrint:NSApplicationDelegateReplySuccess];
}

// --- loading --------------------------------------------------------------

- (void)loadPaths:(NSArray<NSString *> *)paths {
    if (_loading || paths.count == 0) return;
    _loading = YES;
    _spinner.hidden = NO;
    [_spinner startAnimation:nil];

    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        std::vector<Entry>              newEntries;
        std::vector<viewer::PointCloud> newClouds;

        for (NSString *path in paths) {
            const std::string p = path.UTF8String;
            const std::string base = [path lastPathComponent].UTF8String;

            dispatch_async(dispatch_get_main_queue(), ^{
                self->_status.stringValue = [NSString stringWithFormat:@"Reading %@…", [path lastPathComponent]];
            });

            e57::Reader reader;
            std::string err;
            if (!reader.open(p, err)) {
                Entry e;
                e.file = base; e.path = p; e.scan = "—";
                e.kind = check::Kind::Unified;
                e.status = "unreadable";
                e.detail = err;
                newEntries.push_back(e);
                continue;
            }

            for (size_t i = 0; i < reader.scanCount(); ++i) {
                Entry e;
                e.file = base;
                e.path = p;
                e.scan = reader.scan(i).name.empty()
                       ? ("scan " + std::to_string(i)) : reader.scan(i).name;
                e.sourcePoints = reader.scan(i).recordCount;

                const check::Result res = check::classify(reader, i);
                e.kind   = res.kind;
                e.status = res.summary;
                for (const auto &line : res.evidence) {
                    if (!e.detail.empty()) e.detail += "\n";
                    e.detail += line;
                }

                if (res.usable()) {
                    viewer::PointCloud pc;
                    std::string lerr;
                    if (viewer::loadCloud(reader, i, viewer::LoadOptions{}, pc, lerr)) {
                        e.loadedPoints = pc.pointCount();
                        e.rendered = true;
                        newClouds.push_back(std::move(pc));
                    } else {
                        e.kind = check::Kind::Unified;
                        e.status = "decode failed";
                        e.detail = lerr;
                    }
                }
                newEntries.push_back(e);
            }
        }

        dispatch_async(dispatch_get_main_queue(), ^{
            for (auto &e : newEntries) self->_entries.push_back(e);
            for (auto &c : newClouds)  self->_clouds.push_back(std::move(c));

            [self->_table reloadData];
            [self->_cloudView setScene:self->_clouds];

            size_t usable = 0, rejected = 0, pts = 0;
            uint64_t src = 0;
            for (const auto &e : self->_entries) {
                if (e.rendered) { ++usable; pts += e.loadedPoints; src += e.sourcePoints; }
                else ++rejected;
            }
            self->_status.stringValue = [NSString stringWithFormat:
                @"%zu scan%s shown (%s of %s points)   ·   %zu rejected", usable,
                usable == 1 ? "" : "s",
                humanCount(pts).c_str(), humanCount(src).c_str(), rejected];

            self->_loading = NO;
            [self->_spinner stopAnimation:nil];
            self->_spinner.hidden = YES;
        });
    });
}

// --- viewer callbacks -----------------------------------------------------

- (void)cloudViewDidChangeView:(NSString *)status {
    // Only overwrite the transient half of the status line; the load summary
    // stays useful while navigating.
    if (!_loading) _status.stringValue = status;
}

// --- table ----------------------------------------------------------------

- (NSInteger)numberOfRowsInTableView:(NSTableView *)tableView {
    (void)tableView;
    return (NSInteger)_entries.size();
}

- (NSView *)tableView:(NSTableView *)tableView
   viewForTableColumn:(NSTableColumn *)column
                  row:(NSInteger)row {
    (void)tableView;
    if (row < 0 || (size_t)row >= _entries.size()) return nil;
    const Entry &e = _entries[(size_t)row];

    NSTextField *label = [[NSTextField alloc] initWithFrame:NSMakeRect(0, 0, column.width, 30)];
    label.bezeled = NO; label.editable = NO; label.drawsBackground = NO;
    label.lineBreakMode = NSLineBreakByTruncatingMiddle;
    label.font = [NSFont systemFontOfSize:11];

    if ([column.identifier isEqualToString:@"scan"]) {
        label.stringValue = ns(e.file + "  ▸  " + e.scan);
        label.toolTip = ns(e.path);
    } else if ([column.identifier isEqualToString:@"status"]) {
        label.stringValue = ns(e.status);
        label.toolTip = ns(e.detail.empty() ? e.status : e.detail);
        switch (e.kind) {
        case check::Kind::Structured: label.textColor = [NSColor systemGreenColor]; break;
        case check::Kind::Unified:    label.textColor = [NSColor systemRedColor];   break;
        case check::Kind::Ambiguous:  label.textColor = [NSColor systemOrangeColor];break;
        }
    } else {
        label.stringValue = e.rendered
            ? ns(humanCount(e.loadedPoints) + " / " + humanCount(e.sourcePoints))
            : ns("—");
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
        AppDelegate *delegate = [[AppDelegate alloc] init];
        app.delegate = delegate;
        [app run];
    }
    return 0;
}
