# CMake generated Testfile for 
# Source directory: /home/vboxuser/Desktop/IRT/eval_k_means_coresets_main/cmake-3.20.2/Utilities/cmcurl
# Build directory: /home/vboxuser/Desktop/IRT/eval_k_means_coresets_main/cmake-3.20.2/Utilities/cmcurl
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(curl "curltest" "http://open.cdash.org/user.php")
set_tests_properties(curl PROPERTIES  _BACKTRACE_TRIPLES "/home/vboxuser/Desktop/IRT/eval_k_means_coresets_main/cmake-3.20.2/Utilities/cmcurl/CMakeLists.txt;1468;add_test;/home/vboxuser/Desktop/IRT/eval_k_means_coresets_main/cmake-3.20.2/Utilities/cmcurl/CMakeLists.txt;0;")
subdirs("lib")
