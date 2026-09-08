import itertools
ctr = itertools.count(1)
def uid(): return "E57C%016X%04X" % (0, next(ctr))

# ---- files: (group, path, filetype, is_source)
GROUPS = {
 "src": [("e57.h","sourcecode.c.h"),("e57.cpp","sourcecode.cpp.cpp"),
         ("math3d.h","sourcecode.c.h"),("camera.h","sourcecode.c.h"),
         ("camera.cpp","sourcecode.cpp.cpp"),("scan_check.h","sourcecode.c.h"),
         ("scan_check.cpp","sourcecode.cpp.cpp"),("point_cloud.h","sourcecode.c.h"),
         ("point_cloud.cpp","sourcecode.cpp.cpp"),("picker.h","sourcecode.c.h"),
         ("picker.cpp","sourcecode.cpp.cpp"),("frame.h","sourcecode.c.h"),
         ("frame.cpp","sourcecode.cpp.cpp"),("lod.h","sourcecode.c.h"),
         ("lod.cpp","sourcecode.cpp.cpp"),("point_store.h","sourcecode.c.h"),
         ("point_store.cpp","sourcecode.cpp.cpp"),("indexer.h","sourcecode.c.h"),
         ("indexer.cpp","sourcecode.cpp.cpp"),("range_image.h","sourcecode.c.h"),
         ("range_image.cpp","sourcecode.cpp.cpp"),("carve.h","sourcecode.c.h"),
         ("carve.cpp","sourcecode.cpp.cpp"),("visibility.h","sourcecode.c.h"),
         ("visibility.cpp","sourcecode.cpp.cpp"),("main.cpp","sourcecode.cpp.cpp")],
 "app": [("CarveGpu.h","sourcecode.c.h"),("CarveGpu.mm","sourcecode.cpp.objcpp"),
         ("Renderer.h","sourcecode.c.h"),("Renderer.mm","sourcecode.cpp.objcpp"),
         ("CloudView.h","sourcecode.c.h"),("CloudView.mm","sourcecode.cpp.objcpp"),
         ("AppDelegate.mm","sourcecode.cpp.objcpp"),("Info.plist","text.plist.xml")],
 "tests": [("e57_fixture.h","sourcecode.c.h"),("test_e57.cpp","sourcecode.cpp.cpp"),
           ("test_viewer.cpp","sourcecode.cpp.cpp"),
           ("test_lod.cpp","sourcecode.cpp.cpp"),
           ("test_indexer.cpp","sourcecode.cpp.cpp"),
           ("test_range_image.cpp","sourcecode.cpp.cpp"),
           ("test_carve.cpp","sourcecode.cpp.cpp"),
           ("test_visibility.cpp","sourcecode.cpp.cpp")],
}
DOCS = [("README.md","net.daringfireball.markdown"),("DESIGN.md","net.daringfireball.markdown"),
        ("CLAUDE.md","net.daringfireball.markdown"),("CMakeLists.txt","text")]

fref = {}
for g, files in GROUPS.items():
    for name, ft in files: fref[(g,name)] = (uid(), ft)
for name, ft in DOCS: fref[(".",name)] = (uid(), ft)

CORE = [("src","e57.cpp"),("src","camera.cpp"),("src","scan_check.cpp"),
        ("src","point_cloud.cpp"),("src","picker.cpp"),("src","frame.cpp"),
        ("src","lod.cpp"),("src","point_store.cpp"),("src","indexer.cpp"),
        ("src","range_image.cpp"),("src","carve.cpp"),("src","visibility.cpp")]
TARGETS = [
  ("e57cov","com.apple.product-type.tool","e57cov","compiled.mach-o-executable",
   CORE+[("src","main.cpp")], False),
  ("test_e57","com.apple.product-type.tool","test_e57","compiled.mach-o-executable",
   CORE+[("tests","test_e57.cpp")], False),
  ("test_viewer","com.apple.product-type.tool","test_viewer","compiled.mach-o-executable",
   CORE+[("tests","test_viewer.cpp")], False),
  ("test_lod","com.apple.product-type.tool","test_lod","compiled.mach-o-executable",
   CORE+[("tests","test_lod.cpp")], False),
  ("test_indexer","com.apple.product-type.tool","test_indexer","compiled.mach-o-executable",
   CORE+[("tests","test_indexer.cpp")], False),
  ("test_range_image","com.apple.product-type.tool","test_range_image","compiled.mach-o-executable",
   CORE+[("tests","test_range_image.cpp")], False),
  ("test_carve","com.apple.product-type.tool","test_carve","compiled.mach-o-executable",
   CORE+[("tests","test_carve.cpp")], False),
  ("test_visibility","com.apple.product-type.tool","test_visibility","compiled.mach-o-executable",
   CORE+[("tests","test_visibility.cpp")], False),
  ("E57CoverageChecker","com.apple.product-type.application","E57CoverageChecker.app",
   "wrapper.application",
   CORE+[("app","Renderer.mm"),("app","CloudView.mm"),("app","CarveGpu.mm"),
         ("app","AppDelegate.mm")], True),
]

prod, tgt, srcphase, cfglist, cfgs, bfiles = {}, {}, {}, {}, {}, {}
for name, ptype, pname, pft, srcs, isapp in TARGETS:
    prod[name]=(uid(), pname, pft); tgt[name]=uid(); srcphase[name]=uid()
    cfglist[name]=uid(); cfgs[name]=(uid(), uid())
    bfiles[name]=[(uid(), fref[s][0], s[1]) for s in srcs]
projcfglist, projcfgs = uid(), (uid(), uid())
mainGroup, productsGroup, docsGroup = uid(), uid(), uid()
groupIds = {g: uid() for g in GROUPS}
rootObj = uid()

L=[]
w=L.append
w("// !$*UTF8*$!\n{\n\tarchiveVersion = 1;\n\tclasses = {\n\t};\n\tobjectVersion = 56;\n\tobjects = {\n")

w("/* Begin PBXBuildFile section */")
for name,_,_,_,_,_ in TARGETS:
    for bid, fid, fname in bfiles[name]:
        w("\t\t%s /* %s in Sources */ = {isa = PBXBuildFile; fileRef = %s /* %s */; };" % (bid,fname,fid,fname))
w("/* End PBXBuildFile section */\n")

w("/* Begin PBXFileReference section */")
for (g,name),(fid,ft) in fref.items():
    w('\t\t%s /* %s */ = {isa = PBXFileReference; lastKnownFileType = %s; path = %s; sourceTree = "<group>"; };' % (fid,name,ft,name))
for name,_,_,_,_,_ in TARGETS:
    pid,pname,pft = prod[name]
    w('\t\t%s /* %s */ = {isa = PBXFileReference; explicitFileType = "%s"; includeInIndex = 0; path = %s; sourceTree = BUILT_PRODUCTS_DIR; };' % (pid,pname,pft,pname))
w("/* End PBXFileReference section */\n")

w("/* Begin PBXGroup section */")
w("\t\t%s = {\n\t\t\tisa = PBXGroup;\n\t\t\tchildren = (" % mainGroup)
for g in ["src","app","tests"]: w("\t\t\t\t%s /* %s */," % (groupIds[g], g))
w("\t\t\t\t%s /* Documentation */," % docsGroup)
w("\t\t\t\t%s /* Products */," % productsGroup)
w('\t\t\t);\n\t\t\tsourceTree = "<group>";\n\t\t};')
for g, files in GROUPS.items():
    w("\t\t%s /* %s */ = {\n\t\t\tisa = PBXGroup;\n\t\t\tchildren = (" % (groupIds[g], g))
    for name,_ in files: w("\t\t\t\t%s /* %s */," % (fref[(g,name)][0], name))
    w('\t\t\t);\n\t\t\tpath = %s;\n\t\t\tsourceTree = "<group>";\n\t\t};' % g)
w("\t\t%s /* Documentation */ = {\n\t\t\tisa = PBXGroup;\n\t\t\tchildren = (" % docsGroup)
for name,_ in DOCS: w("\t\t\t\t%s /* %s */," % (fref[(".",name)][0], name))
w('\t\t\t);\n\t\t\tname = Documentation;\n\t\t\tsourceTree = "<group>";\n\t\t};')
w("\t\t%s /* Products */ = {\n\t\t\tisa = PBXGroup;\n\t\t\tchildren = (" % productsGroup)
for name,_,_,_,_,_ in TARGETS: w("\t\t\t\t%s /* %s */," % (prod[name][0], prod[name][1]))
w('\t\t\t);\n\t\t\tname = Products;\n\t\t\tsourceTree = "<group>";\n\t\t};')
w("/* End PBXGroup section */\n")

w("/* Begin PBXNativeTarget section */")
for name,ptype,pname,_,_,_ in TARGETS:
    w("""\t\t%s /* %s */ = {
\t\t\tisa = PBXNativeTarget;
\t\t\tbuildConfigurationList = %s /* Build configuration list for PBXNativeTarget "%s" */;
\t\t\tbuildPhases = (
\t\t\t\t%s /* Sources */,
\t\t\t);
\t\t\tbuildRules = (
\t\t\t);
\t\t\tdependencies = (
\t\t\t);
\t\t\tname = %s;
\t\t\tproductName = %s;
\t\t\tproductReference = %s /* %s */;
\t\t\tproductType = "%s";
\t\t};""" % (tgt[name],name,cfglist[name],name,srcphase[name],name,name,prod[name][0],pname,ptype))
w("/* End PBXNativeTarget section */\n")

w("""/* Begin PBXProject section */
\t\t%s /* Project object */ = {
\t\t\tisa = PBXProject;
\t\t\tattributes = {
\t\t\t\tBuildIndependentTargetsInParallel = 1;
\t\t\t\tLastUpgradeCheck = 1540;
\t\t\t\tTargetAttributes = {""" % rootObj)
for name,_,_,_,_,_ in TARGETS:
    w("\t\t\t\t\t%s = {\n\t\t\t\t\t\tCreatedOnToolsVersion = 15.4;\n\t\t\t\t\t};" % tgt[name])
w("""\t\t\t\t};
\t\t\t};
\t\t\tbuildConfigurationList = %s /* Build configuration list for PBXProject "E57CoverageChecker" */;
\t\t\tcompatibilityVersion = "Xcode 14.0";
\t\t\tdevelopmentRegion = en;
\t\t\thasScannedForEncodings = 0;
\t\t\tknownRegions = (
\t\t\t\ten,
\t\t\t\tBase,
\t\t\t);
\t\t\tmainGroup = %s;
\t\t\tproductRefGroup = %s /* Products */;
\t\t\tprojectDirPath = "";
\t\t\tprojectRoot = "";
\t\t\ttargets = (""" % (projcfglist, mainGroup, productsGroup))
for name,_,_,_,_,_ in TARGETS: w("\t\t\t\t%s /* %s */," % (tgt[name], name))
w("\t\t\t);\n\t\t};\n/* End PBXProject section */\n")

w("/* Begin PBXSourcesBuildPhase section */")
for name,_,_,_,_,_ in TARGETS:
    w("\t\t%s /* Sources */ = {\n\t\t\tisa = PBXSourcesBuildPhase;\n\t\t\tbuildActionMask = 2147483647;\n\t\t\tfiles = (" % srcphase[name])
    for bid,_,fname in bfiles[name]: w("\t\t\t\t%s /* %s in Sources */," % (bid,fname))
    w("\t\t\t);\n\t\t\trunOnlyForDeploymentPostprocessing = 0;\n\t\t};")
w("/* End PBXSourcesBuildPhase section */\n")

COMMON = """\t\t\t\tALWAYS_SEARCH_USER_PATHS = NO;
\t\t\t\tCLANG_ANALYZER_NONNULL = YES;
\t\t\t\tCLANG_CXX_LANGUAGE_STANDARD = "c++20";
\t\t\t\tCLANG_CXX_LIBRARY = "libc++";
\t\t\t\tCLANG_ENABLE_MODULES = YES;
\t\t\t\tCLANG_ENABLE_OBJC_ARC = YES;
\t\t\t\tCLANG_WARN_BOOL_CONVERSION = YES;
\t\t\t\tCLANG_WARN_DOCUMENTATION_COMMENTS = YES;
\t\t\t\tCLANG_WARN_EMPTY_BODY = YES;
\t\t\t\tCLANG_WARN_ENUM_CONVERSION = YES;
\t\t\t\tCLANG_WARN_INFINITE_RECURSION = YES;
\t\t\t\tCLANG_WARN_INT_CONVERSION = YES;
\t\t\t\tCLANG_WARN_OBJC_ROOT_CLASS = YES_ERROR;
\t\t\t\tCLANG_WARN_RANGE_LOOP_ANALYSIS = YES;
\t\t\t\tCLANG_WARN_SUSPICIOUS_MOVE = YES;
\t\t\t\tCLANG_WARN_UNREACHABLE_CODE = YES;
\t\t\t\tCOPY_PHASE_STRIP = NO;
\t\t\t\tENABLE_STRICT_OBJC_MSGSEND = YES;
\t\t\t\tGCC_C_LANGUAGE_STANDARD = gnu17;
\t\t\t\tGCC_NO_COMMON_BLOCKS = YES;
\t\t\t\tGCC_WARN_ABOUT_RETURN_TYPE = YES;
\t\t\t\tGCC_WARN_UNINITIALIZED_AUTOS = YES;
\t\t\t\tGCC_WARN_UNUSED_FUNCTION = YES;
\t\t\t\tGCC_WARN_UNUSED_VARIABLE = YES;
\t\t\t\tMACOSX_DEPLOYMENT_TARGET = 14.0;
\t\t\t\tMTL_FAST_MATH = YES;
\t\t\t\tSDKROOT = macosx;
"""
w("/* Begin XCBuildConfiguration section */")
w("\t\t%s /* Debug */ = {\n\t\t\tisa = XCBuildConfiguration;\n\t\t\tbuildSettings = {\n%s\t\t\t\tDEBUG_INFORMATION_FORMAT = dwarf;\n\t\t\t\tENABLE_TESTABILITY = YES;\n\t\t\t\tGCC_DYNAMIC_NO_PIC = NO;\n\t\t\t\tGCC_OPTIMIZATION_LEVEL = 0;\n\t\t\t\tGCC_PREPROCESSOR_DEFINITIONS = (\n\t\t\t\t\t\"DEBUG=1\",\n\t\t\t\t\t\"$(inherited)\",\n\t\t\t\t);\n\t\t\t\tMTL_ENABLE_DEBUG_INFO = INCLUDE_SOURCE;\n\t\t\t\tONLY_ACTIVE_ARCH = YES;\n\t\t\t};\n\t\t\tname = Debug;\n\t\t};" % (projcfgs[0], COMMON))
w("\t\t%s /* Release */ = {\n\t\t\tisa = XCBuildConfiguration;\n\t\t\tbuildSettings = {\n%s\t\t\t\tDEBUG_INFORMATION_FORMAT = \"dwarf-with-dsym\";\n\t\t\t\tENABLE_NS_ASSERTIONS = NO;\n\t\t\t\tGCC_OPTIMIZATION_LEVEL = 3;\n\t\t\t\tMTL_ENABLE_DEBUG_INFO = NO;\n\t\t\t};\n\t\t\tname = Release;\n\t\t};" % (projcfgs[1], COMMON))

for name,ptype,pname,_,_,isapp in TARGETS:
    extra = ""
    if isapp:
        extra = ('\t\t\t\tINFOPLIST_FILE = app/Info.plist;\n'
                 '\t\t\t\tPRODUCT_BUNDLE_IDENTIFIER = "com.mjankor.e57coveragechecker";\n'
                 '\t\t\t\tCOMBINE_HIDPI_IMAGES = YES;\n'
                 '\t\t\t\tLD_RUNPATH_SEARCH_PATHS = (\n\t\t\t\t\t"$(inherited)",\n\t\t\t\t\t"@executable_path/../Frameworks",\n\t\t\t\t);\n'
                 '\t\t\t\tOTHER_LDFLAGS = (\n'
                 '\t\t\t\t\t"-framework", "Cocoa",\n'
                 '\t\t\t\t\t"-framework", "Metal",\n'
                 '\t\t\t\t\t"-framework", "MetalKit",\n'
                 '\t\t\t\t\t"-framework", "QuartzCore",\n'
                 '\t\t\t\t\t"-framework", "UniformTypeIdentifiers",\n'
                 '\t\t\t\t);\n')
    for cfg, cname in ((cfgs[name][0], "Debug"), (cfgs[name][1], "Release")):
        w('\t\t%s /* %s */ = {\n\t\t\tisa = XCBuildConfiguration;\n\t\t\tbuildSettings = {\n'
          '\t\t\t\tCODE_SIGN_IDENTITY = "-";\n\t\t\t\tCODE_SIGN_STYLE = Automatic;\n'
          '\t\t\t\tHEADER_SEARCH_PATHS = "$(SRCROOT)/src";\n'
          '\t\t\t\tPRODUCT_NAME = "$(TARGET_NAME)";\n'
          '%s'
          '\t\t\t\tWARNING_CFLAGS = (\n\t\t\t\t\t"-Wall",\n\t\t\t\t\t"-Wextra",\n\t\t\t\t);\n'
          '\t\t\t};\n\t\t\tname = %s;\n\t\t};' % (cfg, cname, extra, cname))
w("/* End XCBuildConfiguration section */\n")

w("/* Begin XCConfigurationList section */")
w('\t\t%s /* Build configuration list for PBXProject "E57CoverageChecker" */ = {\n\t\t\tisa = XCConfigurationList;\n\t\t\tbuildConfigurations = (\n\t\t\t\t%s /* Debug */,\n\t\t\t\t%s /* Release */,\n\t\t\t);\n\t\t\tdefaultConfigurationIsVisible = 0;\n\t\t\tdefaultConfigurationName = Release;\n\t\t};' % (projcfglist, projcfgs[0], projcfgs[1]))
for name,_,_,_,_,_ in TARGETS:
    w('\t\t%s /* Build configuration list for PBXNativeTarget "%s" */ = {\n\t\t\tisa = XCConfigurationList;\n\t\t\tbuildConfigurations = (\n\t\t\t\t%s /* Debug */,\n\t\t\t\t%s /* Release */,\n\t\t\t);\n\t\t\tdefaultConfigurationIsVisible = 0;\n\t\t\tdefaultConfigurationName = Release;\n\t\t};' % (cfglist[name], name, cfgs[name][0], cfgs[name][1]))
w("/* End XCConfigurationList section */")

w("\t};\n\trootObject = %s /* Project object */;\n}" % rootObj)

open("/workspace/e57-coverage-checker/E57CoverageChecker.xcodeproj/project.pbxproj","w").write("\n".join(L)+"\n")

# Schemes
SCHEME = '''<?xml version="1.0" encoding="UTF-8"?>
<Scheme LastUpgradeVersion = "1540" version = "1.7">
   <BuildAction parallelizeBuildables = "YES" buildImplicitDependencies = "YES">
      <BuildActionEntries>
         <BuildActionEntry buildForTesting = "YES" buildForRunning = "YES" buildForProfiling = "YES" buildForArchiving = "YES" buildForAnalyzing = "YES">
            <BuildableReference BuildableIdentifier = "primary" BlueprintIdentifier = "%(id)s" BuildableName = "%(prod)s" BlueprintName = "%(name)s" ReferencedContainer = "container:E57CoverageChecker.xcodeproj">
            </BuildableReference>
         </BuildActionEntry>
      </BuildActionEntries>
   </BuildAction>
   <TestAction buildConfiguration = "Debug" selectedDebuggerIdentifier = "Xcode.DebuggerFoundation.Debugger.LLDB" selectedLauncherIdentifier = "Xcode.DebuggerFoundation.Launcher.LLDB" shouldUseLaunchSchemeArgsEnv = "YES">
      <Testables>
      </Testables>
   </TestAction>
   <LaunchAction buildConfiguration = "Debug" selectedDebuggerIdentifier = "Xcode.DebuggerFoundation.Debugger.LLDB" selectedLauncherIdentifier = "Xcode.DebuggerFoundation.Launcher.LLDB" launchStyle = "0" useCustomWorkingDirectory = "NO" ignoresPersistentStateOnLaunch = "NO" debugDocumentVersioning = "YES" debugServiceExtension = "internal" allowLocationSimulation = "YES">
      <BuildableProductRunnable runnableDebuggingMode = "0">
         <BuildableReference BuildableIdentifier = "primary" BlueprintIdentifier = "%(id)s" BuildableName = "%(prod)s" BlueprintName = "%(name)s" ReferencedContainer = "container:E57CoverageChecker.xcodeproj">
         </BuildableReference>
      </BuildableProductRunnable>
   </LaunchAction>
   <ProfileAction buildConfiguration = "Release" shouldUseLaunchSchemeArgsEnv = "YES" savedToolIdentifier = "" useCustomWorkingDirectory = "NO" debugDocumentVersioning = "YES">
      <BuildableProductRunnable runnableDebuggingMode = "0">
         <BuildableReference BuildableIdentifier = "primary" BlueprintIdentifier = "%(id)s" BuildableName = "%(prod)s" BlueprintName = "%(name)s" ReferencedContainer = "container:E57CoverageChecker.xcodeproj">
         </BuildableReference>
      </BuildableProductRunnable>
   </ProfileAction>
   <AnalyzeAction buildConfiguration = "Debug">
   </AnalyzeAction>
   <ArchiveAction buildConfiguration = "Release" revealArchiveInOrganizer = "YES">
   </ArchiveAction>
</Scheme>
'''
import os
sd="/workspace/e57-coverage-checker/E57CoverageChecker.xcodeproj/xcshareddata/xcschemes"
for f in os.listdir(sd): os.remove(os.path.join(sd,f))
for name,_,pname,_,_,_ in TARGETS:
    open(os.path.join(sd,name+".xcscheme"),"w").write(
        SCHEME % {"id": tgt[name], "prod": pname, "name": name})
print("generated", len(TARGETS), "targets")
