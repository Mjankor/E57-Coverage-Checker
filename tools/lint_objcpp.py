#!/usr/bin/env python3
"""Checks on app/*.mm that nothing else here can make.

The library and the tests build under CMake on any platform, so a mistake in them
is caught in seconds. The app is Objective-C++ against AppKit and Metal and builds
only in Xcode on a Mac, so a mistake in it is caught by whoever next opens Xcode —
which in practice has meant shipping compile errors and finding out later.

This does not compile anything. It looks for the two mistakes that have actually
been made, both of which are mechanical:

  1. A selector sent with @selector() that nothing defines. A typo here compiles
     and then raises doesNotRecognizeSelector at the moment the menu item is
     clicked, which is the worst time to find out.

  2. A C++ container captured by a block and mutated inside it. A block captures
     by value and the capture is CONST, so `v.push_back(...)` on a captured vector
     does not compile — but it reads exactly like code that should work, and the
     error names push_back rather than the capture.

Both are narrow and neither is clever. Run it after editing app/.

    python3 tools/lint_objcpp.py app
"""

import re
import sys
from pathlib import Path

# Selectors AppKit, the protocols this app adopts, and the runtime provide. Not a
# complete list of Cocoa — only what this app actually sends — because the point is
# to catch a typo in OUR selectors, and an unknown framework selector here would be
# noise rather than a finding.
FRAMEWORK = {
    "terminate:", "openDocument:", "copy:", "paste:", "performClose:",
    "tableViewSelectionDidChange:", "numberOfRowsInTableView:",
    "tableView:viewForTableColumn:row:", "windowDidResize:",
    "applicationDidFinishLaunching:", "applicationShouldTerminateAfterLastWindowClosed:",
    "validateMenuItem:", "resetCursorRects", "mouseDragged:", "dealloc", "init",
}

MUTATORS = ("push_back", "emplace_back", "emplace", "insert", "erase", "clear",
            "resize", "assign", "append", "pop_back")


def method_names(src: str) -> set:
    """Selectors defined in this file, as full selector strings.

    DOTALL on the signature, because an Objective-C method signature wraps across
    lines as often as not and a line-at-a-time match simply does not see those —
    which this checker got wrong first time out, reporting a perfectly good
    three-line delegate method as undefined. `[^;{]*` is bounded by the brace that
    opens the body, and a signature can hold neither character.
    """
    out = set()
    for m in re.finditer(r"^[ \t]*[-+][ \t]*\([^)]*\)([^;{]*)\{", src, re.M | re.S):
        sig = m.group(1)
        parts = re.findall(r"([A-Za-z_]\w*)\s*:", sig)
        if parts:
            out.add("".join(p + ":" for p in parts))
        else:
            name = sig.strip().split()[0] if sig.strip() else ""
            if name:
                out.add(name)
    return out


def selector_check(files) -> list:
    defined, sent = set(), {}
    for f in files:
        src = f.read_text()
        defined |= {s.split(":")[0] for s in method_names(src)}
        for m in re.finditer(r"@selector\(([^)]+)\)", src):
            sent.setdefault(m.group(1).split(":")[0], (f, src[:m.start()].count("\n") + 1))
    bad = []
    for name, (f, line) in sorted(sent.items()):
        if name in defined:
            continue
        if any(fw.split(":")[0] == name for fw in FRAMEWORK):
            continue
        bad.append(f"{f}:{line}  @selector({name}) is not defined in app/")
    return bad


def block_capture_check(files) -> list:
    """C++ containers captured by a block and mutated inside it."""
    bad = []
    for f in files:
        lines = f.read_text().split("\n")
        # Names that are safe to mutate inside a block: declared __block, declared
        # inside the block itself, or reached through self.
        blocked, local, depth, start = set(), set(), 0, 0
        for i, line in enumerate(lines, 1):
            m = re.search(r"__block\s+[\w:<>,\s*&]*?\b([A-Za-z_]\w*)\s*[=;]", line)
            if m:
                blocked.add(m.group(1))
            if re.search(r"\^\s*(\([^)]*\))?\s*\{", line):
                depth, start, local = 1, i, set()
                continue
            if not depth:
                continue
            depth += line.count("{") - line.count("}")
            # A declaration inside the block is the block's own variable.
            d = re.search(r"^\s*(?:const\s+)?std::\w+<[^;]*>\s+([A-Za-z_]\w*)\s*[;=({]", line)
            if d:
                local.add(d.group(1))
            for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\.\s*(" + "|".join(MUTATORS) + r")\s*\(", line):
                name = m.group(1)
                if name in blocked or name in local or name.startswith("_"):
                    continue
                if "self->" in line or "->" + name in line:
                    continue
                bad.append(f"{f}:{i}  '{name}.{m.group(2)}()' inside the block opened at "
                           f"line {start}: a captured C++ object is const — declare it "
                           f"__block, or build it outside the block")
            if depth <= 0:
                depth = 0
    return bad


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "app")
    files = sorted(root.glob("*.mm"))
    if not files:
        print(f"no .mm files under {root}")
        return 2

    problems = selector_check(files) + block_capture_check(files)
    print(f"{root}: {len(files)} file(s), {sum(len(f.read_text().split(chr(10))) for f in files)} lines")
    for p in problems:
        print(f"  ERROR {p}")
    print("  OK" if not problems else f"  {len(problems)} PROBLEM(S)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
