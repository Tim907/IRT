# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/vboxuser/Desktop/IRT/eval_k_means_coresets_main/cmake-3.20.2/Bootstrap.cmk/cmake

# The command to remove a file.
RM = /home/vboxuser/Desktop/IRT/eval_k_means_coresets_main/cmake-3.20.2/Bootstrap.cmk/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/vboxuser/Desktop/IRT/eval_k_means_coresets_main/cmake-3.20.2/Source/Checks/Curses

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/vboxuser/Desktop/IRT/eval_k_means_coresets_main/cmake-3.20.2/Source/Checks/Curses-build

# Include any dependencies generated for this target.
include CMakeFiles/CheckCurses.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/CheckCurses.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/CheckCurses.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/CheckCurses.dir/flags.make

CMakeFiles/CheckCurses.dir/CheckCurses.c.o: CMakeFiles/CheckCurses.dir/flags.make
CMakeFiles/CheckCurses.dir/CheckCurses.c.o: /home/vboxuser/Desktop/IRT/eval_k_means_coresets_main/cmake-3.20.2/Source/Checks/Curses/CheckCurses.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --progress-dir=/home/vboxuser/Desktop/IRT/eval_k_means_coresets_main/cmake-3.20.2/Source/Checks/Curses-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/CheckCurses.dir/CheckCurses.c.o"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/CheckCurses.dir/CheckCurses.c.o -c /home/vboxuser/Desktop/IRT/eval_k_means_coresets_main/cmake-3.20.2/Source/Checks/Curses/CheckCurses.c

CMakeFiles/CheckCurses.dir/CheckCurses.c.i: cmake_force
	@echo "Preprocessing C source to CMakeFiles/CheckCurses.dir/CheckCurses.c.i"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/vboxuser/Desktop/IRT/eval_k_means_coresets_main/cmake-3.20.2/Source/Checks/Curses/CheckCurses.c > CMakeFiles/CheckCurses.dir/CheckCurses.c.i

CMakeFiles/CheckCurses.dir/CheckCurses.c.s: cmake_force
	@echo "Compiling C source to assembly CMakeFiles/CheckCurses.dir/CheckCurses.c.s"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/vboxuser/Desktop/IRT/eval_k_means_coresets_main/cmake-3.20.2/Source/Checks/Curses/CheckCurses.c -o CMakeFiles/CheckCurses.dir/CheckCurses.c.s

# Object files for target CheckCurses
CheckCurses_OBJECTS = \
"CMakeFiles/CheckCurses.dir/CheckCurses.c.o"

# External object files for target CheckCurses
CheckCurses_EXTERNAL_OBJECTS =

CheckCurses: CMakeFiles/CheckCurses.dir/CheckCurses.c.o
CheckCurses: CMakeFiles/CheckCurses.dir/build.make
CheckCurses: /usr/lib/x86_64-linux-gnu/libncurses.so
CheckCurses: /usr/lib/x86_64-linux-gnu/libform.so
CheckCurses: CMakeFiles/CheckCurses.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --progress-dir=/home/vboxuser/Desktop/IRT/eval_k_means_coresets_main/cmake-3.20.2/Source/Checks/Curses-build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable CheckCurses"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CheckCurses.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/CheckCurses.dir/build: CheckCurses
.PHONY : CMakeFiles/CheckCurses.dir/build

CMakeFiles/CheckCurses.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/CheckCurses.dir/cmake_clean.cmake
.PHONY : CMakeFiles/CheckCurses.dir/clean

CMakeFiles/CheckCurses.dir/depend:
	cd /home/vboxuser/Desktop/IRT/eval_k_means_coresets_main/cmake-3.20.2/Source/Checks/Curses-build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vboxuser/Desktop/IRT/eval_k_means_coresets_main/cmake-3.20.2/Source/Checks/Curses /home/vboxuser/Desktop/IRT/eval_k_means_coresets_main/cmake-3.20.2/Source/Checks/Curses /home/vboxuser/Desktop/IRT/eval_k_means_coresets_main/cmake-3.20.2/Source/Checks/Curses-build /home/vboxuser/Desktop/IRT/eval_k_means_coresets_main/cmake-3.20.2/Source/Checks/Curses-build /home/vboxuser/Desktop/IRT/eval_k_means_coresets_main/cmake-3.20.2/Source/Checks/Curses-build/CMakeFiles/CheckCurses.dir/DependInfo.cmake
.PHONY : CMakeFiles/CheckCurses.dir/depend

