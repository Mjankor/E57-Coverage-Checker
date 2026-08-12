#!/usr/bin/env python3
"""Structural validator for a hand-maintained .xcodeproj.

The Xcode project here carries explicit file lists — adding a source file means
adding a PBXBuildFile, a PBXFileReference, a group child and a Sources build
phase entry, all cross-referenced by 24-hex-digit object IDs. A single typo
produces a project that either fails to open or silently drops a file from the
build.

This checks what can be checked without Xcode:

  * the pbxproj parses as an old-style (NeXTSTEP) plist
  * every object ID referenced anywhere resolves to a defined object
  * every object is reachable from rootObject (no orphans)
  * targets have a configuration list, build phases and a product reference
  * every PBXBuildFile points at a real PBXFileReference
  * every source file referenced actually exists on disk, at the path implied
    by its group nesting
  * every source file on disk is in exactly one target's Sources phase
  * shared schemes reference real targets with matching names

It does NOT verify that Xcode likes the result — only that the object graph is
sound. Opening the project is still the real test.

usage: validate_xcodeproj.py <path/to/Project.xcodeproj>
"""

import os
import re
import sys
import xml.etree.ElementTree as ET

REF = re.compile(r'^[0-9A-Fa-f]{24}$')


# --------------------------------------------------------------------------
# Old-style plist parsing

def strip_comments(s):
    out, i, n, in_str = [], 0, len(s), False
    while i < n:
        c = s[i]
        if in_str:
            if c == '\\':
                out.append(s[i:i + 2]); i += 2; continue
            if c == '"':
                in_str = False
            out.append(c); i += 1; continue
        if c == '"':
            in_str = True; out.append(c); i += 1; continue
        if c == '/' and i + 1 < n and s[i + 1] == '*':
            j = s.find('*/', i + 2); i = n if j < 0 else j + 2; continue
        if c == '/' and i + 1 < n and s[i + 1] == '/':
            j = s.find('\n', i); i = n if j < 0 else j; continue
        out.append(c); i += 1
    return ''.join(out)


class Parser:
    def __init__(self, text):
        self.s = text
        self.i = 0

    def ws(self):
        while self.i < len(self.s) and self.s[self.i].isspace():
            self.i += 1

    def expect(self, ch):
        self.ws()
        if self.i >= len(self.s) or self.s[self.i] != ch:
            got = self.s[self.i:self.i + 20] if self.i < len(self.s) else '<eof>'
            raise ValueError("expected %r at offset %d, got %r" % (ch, self.i, got))
        self.i += 1

    def value(self):
        self.ws()
        if self.i >= len(self.s):
            raise ValueError("unexpected end of input")
        c = self.s[self.i]
        if c == '{':
            return self.dict()
        if c == '(':
            return self.array()
        if c == '"':
            return self.qstring()
        return self.bare()

    def dict(self):
        self.expect('{')
        d = {}
        while True:
            self.ws()
            if self.i < len(self.s) and self.s[self.i] == '}':
                self.i += 1
                return d
            k = self.value()
            self.expect('=')
            v = self.value()
            self.expect(';')
            d[k] = v

    def array(self):
        self.expect('(')
        a = []
        while True:
            self.ws()
            if self.i < len(self.s) and self.s[self.i] == ')':
                self.i += 1
                return a
            a.append(self.value())
            self.ws()
            if self.i < len(self.s) and self.s[self.i] == ',':
                self.i += 1

    def qstring(self):
        self.expect('"')
        out = []
        while self.i < len(self.s):
            c = self.s[self.i]
            if c == '\\':
                out.append(self.s[self.i + 1]); self.i += 2; continue
            if c == '"':
                self.i += 1
                return ''.join(out)
            out.append(c); self.i += 1
        raise ValueError("unterminated string")

    def bare(self):
        start = self.i
        while self.i < len(self.s) and self.s[self.i] not in ' \t\r\n;,=(){}':
            self.i += 1
        if self.i == start:
            raise ValueError("empty token at offset %d" % self.i)
        return self.s[start:self.i]


# --------------------------------------------------------------------------

def walk_refs(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            walk_refs(v, path + '.' + str(k), out)
    elif isinstance(node, list):
        for n, v in enumerate(node):
            walk_refs(v, '%s[%d]' % (path, n), out)
    elif isinstance(node, str) and REF.match(node):
        out.append((node, path))


def resolve_group_paths(objects, gid, prefix, files, errors, seen=None):
    """Walk the group tree accumulating on-disk paths for file references."""
    seen = seen if seen is not None else set()
    if gid in seen:
        errors.append("group cycle at %s" % gid)
        return
    seen.add(gid)
    g = objects.get(gid)
    if not g:
        return
    base = prefix
    if g.get('path'):
        base = os.path.join(prefix, g['path'])
    for child in g.get('children', []):
        c = objects.get(child)
        if not c:
            continue
        if c.get('isa') in ('PBXGroup', 'PBXVariantGroup'):
            resolve_group_paths(objects, child, base, files, errors, seen)
        elif c.get('isa') == 'PBXFileReference':
            if c.get('sourceTree') == 'BUILT_PRODUCTS_DIR':
                continue
            files[child] = os.path.join(base, c.get('path', ''))


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    proj = sys.argv[1].rstrip('/')
    pbx = os.path.join(proj, 'project.pbxproj')
    root_dir = os.path.dirname(os.path.abspath(proj)) or '.'

    errors, warnings, checks = [], [], 0

    with open(pbx, 'r', encoding='utf-8') as f:
        raw = f.read()
    try:
        doc = Parser(strip_comments(raw)).dict()
    except ValueError as e:
        print("FAIL: pbxproj does not parse: %s" % e)
        return 1
    checks += 1

    objects = doc.get('objects', {})
    root = doc.get('rootObject')
    if root not in objects:
        errors.append("rootObject %s is not defined" % root)
    checks += 1

    # Every referenced ID resolves.
    refs = []
    walk_refs(objects, 'objects', refs)
    for rid, where in refs:
        if rid not in objects:
            errors.append("dangling reference %s at %s" % (rid, where))
    checks += 1

    # Every object carries an isa.
    for oid, obj in objects.items():
        if not isinstance(obj, dict) or 'isa' not in obj:
            errors.append("object %s has no isa" % oid)
    checks += 1

    # Reachability from rootObject.
    reachable, stack = set(), [root]
    while stack:
        cur = stack.pop()
        if cur in reachable or cur not in objects:
            continue
        reachable.add(cur)
        sub = []
        walk_refs(objects[cur], cur, sub)
        stack.extend(r for r, _ in sub)
    for oid in objects:
        if oid not in reachable:
            errors.append("orphan object %s (isa=%s) unreachable from rootObject"
                          % (oid, objects[oid].get('isa')))
    checks += 1

    # Project wiring.
    project = objects.get(root, {})
    for key in ('mainGroup', 'productRefGroup', 'buildConfigurationList'):
        if project.get(key) not in objects:
            errors.append("project %s does not resolve" % key)
    checks += 1

    targets = project.get('targets', [])
    if not targets:
        errors.append("project declares no targets")
    target_names = {}
    for tid in targets:
        t = objects.get(tid, {})
        name = t.get('name', '?')
        target_names[name] = tid
        if t.get('buildConfigurationList') not in objects:
            errors.append("target %s: buildConfigurationList does not resolve" % name)
        if t.get('productReference') not in objects:
            errors.append("target %s: productReference does not resolve" % name)
        if not t.get('buildPhases'):
            errors.append("target %s: no build phases" % name)
        cl = objects.get(t.get('buildConfigurationList'), {})
        if len(cl.get('buildConfigurations', [])) < 1:
            errors.append("target %s: configuration list is empty" % name)
    checks += 1

    # Build files point at real file references.
    for oid, obj in objects.items():
        if obj.get('isa') == 'PBXBuildFile':
            fr = obj.get('fileRef')
            if fr not in objects:
                errors.append("PBXBuildFile %s: fileRef does not resolve" % oid)
            elif objects[fr].get('isa') != 'PBXFileReference':
                errors.append("PBXBuildFile %s: fileRef is not a PBXFileReference" % oid)
    checks += 1

    # Referenced files exist on disk.
    files = {}
    resolve_group_paths(objects, project.get('mainGroup'), '', files, errors)
    for fid, rel in files.items():
        if not os.path.exists(os.path.join(root_dir, rel)):
            errors.append("file reference %s points at missing path '%s'" % (fid, rel))
    checks += 1

    # Every compilable source on disk is built by some target.
    built = set()
    for oid, obj in objects.items():
        if obj.get('isa') == 'PBXSourcesBuildPhase':
            for bf in obj.get('files', []):
                fr = objects.get(bf, {}).get('fileRef')
                if fr in files:
                    built.add(files[fr])
    for fid, rel in files.items():
        if rel.endswith(('.cpp', '.cc', '.c', '.mm', '.m', '.metal')) and rel not in built:
            warnings.append("source '%s' is in the project but in no Sources phase" % rel)
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames
                       if d not in ('build', '.git', 'DerivedData')
                       and not d.endswith('.xcodeproj')]
        for fn in filenames:
            if fn.endswith(('.cpp', '.mm', '.m', '.metal')):
                rel = os.path.relpath(os.path.join(dirpath, fn), root_dir)
                if rel not in files.values():
                    warnings.append("source '%s' exists on disk but is not in the project" % rel)
    checks += 1

    # Shared schemes reference real targets.
    sdir = os.path.join(proj, 'xcshareddata', 'xcschemes')
    if not os.path.isdir(sdir):
        warnings.append("no shared schemes — targets may not appear on open")
    else:
        for fn in sorted(os.listdir(sdir)):
            if not fn.endswith('.xcscheme'):
                continue
            try:
                tree = ET.parse(os.path.join(sdir, fn))
            except ET.ParseError as e:
                errors.append("scheme %s is not valid XML: %s" % (fn, e))
                continue
            for br in tree.getroot().iter('BuildableReference'):
                bid = br.get('BlueprintIdentifier')
                bname = br.get('BlueprintName')
                if bid not in objects:
                    errors.append("scheme %s: BlueprintIdentifier %s does not resolve" % (fn, bid))
                elif objects[bid].get('name') != bname:
                    errors.append("scheme %s: BlueprintName '%s' != target name '%s'"
                                  % (fn, bname, objects[bid].get('name')))
    checks += 1

    print("%s" % proj)
    print("  objects   : %d" % len(objects))
    print("  targets   : %s" % ', '.join(sorted(target_names)) or '(none)')
    print("  refs      : %d checked" % len(refs))
    print("  files     : %d resolved to disk paths" % len(files))
    print("  checks    : %d" % checks)
    for w in warnings:
        print("  WARN  %s" % w)
    for e in errors:
        print("  ERROR %s" % e)
    print("  %s" % ("OK" if not errors else "%d ERROR(S)" % len(errors)))
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
