# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/kobayasimikio/Desktop/CLionProjects/layer6

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/kobayasimikio/Desktop/CLionProjects/layer6/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/layer6.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/layer6.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/layer6.dir/flags.make

CMakeFiles/layer6.dir/main.c.o: CMakeFiles/layer6.dir/flags.make
CMakeFiles/layer6.dir/main.c.o: ../main.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/kobayasimikio/Desktop/CLionProjects/layer6/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/layer6.dir/main.c.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/layer6.dir/main.c.o   -c /Users/kobayasimikio/Desktop/CLionProjects/layer6/main.c

CMakeFiles/layer6.dir/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/layer6.dir/main.c.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/kobayasimikio/Desktop/CLionProjects/layer6/main.c > CMakeFiles/layer6.dir/main.c.i

CMakeFiles/layer6.dir/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/layer6.dir/main.c.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/kobayasimikio/Desktop/CLionProjects/layer6/main.c -o CMakeFiles/layer6.dir/main.c.s

CMakeFiles/layer6.dir/main.c.o.requires:

.PHONY : CMakeFiles/layer6.dir/main.c.o.requires

CMakeFiles/layer6.dir/main.c.o.provides: CMakeFiles/layer6.dir/main.c.o.requires
	$(MAKE) -f CMakeFiles/layer6.dir/build.make CMakeFiles/layer6.dir/main.c.o.provides.build
.PHONY : CMakeFiles/layer6.dir/main.c.o.provides

CMakeFiles/layer6.dir/main.c.o.provides.build: CMakeFiles/layer6.dir/main.c.o


CMakeFiles/layer6.dir/inference6(hodai16).c.o: CMakeFiles/layer6.dir/flags.make
CMakeFiles/layer6.dir/inference6(hodai16).c.o: ../inference6(hodai16).c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/kobayasimikio/Desktop/CLionProjects/layer6/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/layer6.dir/inference6(hodai16).c.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o "CMakeFiles/layer6.dir/inference6(hodai16).c.o"   -c "/Users/kobayasimikio/Desktop/CLionProjects/layer6/inference6(hodai16).c"

CMakeFiles/layer6.dir/inference6(hodai16).c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/layer6.dir/inference6(hodai16).c.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E "/Users/kobayasimikio/Desktop/CLionProjects/layer6/inference6(hodai16).c" > "CMakeFiles/layer6.dir/inference6(hodai16).c.i"

CMakeFiles/layer6.dir/inference6(hodai16).c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/layer6.dir/inference6(hodai16).c.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S "/Users/kobayasimikio/Desktop/CLionProjects/layer6/inference6(hodai16).c" -o "CMakeFiles/layer6.dir/inference6(hodai16).c.s"

CMakeFiles/layer6.dir/inference6(hodai16).c.o.requires:

.PHONY : CMakeFiles/layer6.dir/inference6(hodai16).c.o.requires

CMakeFiles/layer6.dir/inference6(hodai16).c.o.provides: CMakeFiles/layer6.dir/inference6(hodai16).c.o.requires
	$(MAKE) -f CMakeFiles/layer6.dir/build.make "CMakeFiles/layer6.dir/inference6(hodai16).c.o.provides.build"
.PHONY : CMakeFiles/layer6.dir/inference6(hodai16).c.o.provides

CMakeFiles/layer6.dir/inference6(hodai16).c.o.provides.build: CMakeFiles/layer6.dir/inference6(hodai16).c.o


CMakeFiles/layer6.dir/inference.c.o: CMakeFiles/layer6.dir/flags.make
CMakeFiles/layer6.dir/inference.c.o: ../inference.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/kobayasimikio/Desktop/CLionProjects/layer6/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/layer6.dir/inference.c.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/layer6.dir/inference.c.o   -c /Users/kobayasimikio/Desktop/CLionProjects/layer6/inference.c

CMakeFiles/layer6.dir/inference.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/layer6.dir/inference.c.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/kobayasimikio/Desktop/CLionProjects/layer6/inference.c > CMakeFiles/layer6.dir/inference.c.i

CMakeFiles/layer6.dir/inference.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/layer6.dir/inference.c.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/kobayasimikio/Desktop/CLionProjects/layer6/inference.c -o CMakeFiles/layer6.dir/inference.c.s

CMakeFiles/layer6.dir/inference.c.o.requires:

.PHONY : CMakeFiles/layer6.dir/inference.c.o.requires

CMakeFiles/layer6.dir/inference.c.o.provides: CMakeFiles/layer6.dir/inference.c.o.requires
	$(MAKE) -f CMakeFiles/layer6.dir/build.make CMakeFiles/layer6.dir/inference.c.o.provides.build
.PHONY : CMakeFiles/layer6.dir/inference.c.o.provides

CMakeFiles/layer6.dir/inference.c.o.provides.build: CMakeFiles/layer6.dir/inference.c.o


# Object files for target layer6
layer6_OBJECTS = \
"CMakeFiles/layer6.dir/main.c.o" \
"CMakeFiles/layer6.dir/inference6(hodai16).c.o" \
"CMakeFiles/layer6.dir/inference.c.o"

# External object files for target layer6
layer6_EXTERNAL_OBJECTS =

layer6: CMakeFiles/layer6.dir/main.c.o
layer6: CMakeFiles/layer6.dir/inference6(hodai16).c.o
layer6: CMakeFiles/layer6.dir/inference.c.o
layer6: CMakeFiles/layer6.dir/build.make
layer6: CMakeFiles/layer6.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/kobayasimikio/Desktop/CLionProjects/layer6/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking C executable layer6"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/layer6.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/layer6.dir/build: layer6

.PHONY : CMakeFiles/layer6.dir/build

CMakeFiles/layer6.dir/requires: CMakeFiles/layer6.dir/main.c.o.requires
CMakeFiles/layer6.dir/requires: CMakeFiles/layer6.dir/inference6(hodai16).c.o.requires
CMakeFiles/layer6.dir/requires: CMakeFiles/layer6.dir/inference.c.o.requires

.PHONY : CMakeFiles/layer6.dir/requires

CMakeFiles/layer6.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/layer6.dir/cmake_clean.cmake
.PHONY : CMakeFiles/layer6.dir/clean

CMakeFiles/layer6.dir/depend:
	cd /Users/kobayasimikio/Desktop/CLionProjects/layer6/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/kobayasimikio/Desktop/CLionProjects/layer6 /Users/kobayasimikio/Desktop/CLionProjects/layer6 /Users/kobayasimikio/Desktop/CLionProjects/layer6/cmake-build-debug /Users/kobayasimikio/Desktop/CLionProjects/layer6/cmake-build-debug /Users/kobayasimikio/Desktop/CLionProjects/layer6/cmake-build-debug/CMakeFiles/layer6.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/layer6.dir/depend

