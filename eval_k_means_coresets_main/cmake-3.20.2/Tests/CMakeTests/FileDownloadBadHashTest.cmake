if(NOT "/home/vboxuser/Desktop/IRT/eval_k_means_coresets_main/cmake-3.20.2/Tests/CMakeTests" MATCHES "^/")
  set(slash /)
endif()
set(url "file://${slash}/home/vboxuser/Desktop/IRT/eval_k_means_coresets_main/cmake-3.20.2/Tests/CMakeTests/FileDownloadInput.png")
set(dir "/home/vboxuser/Desktop/IRT/eval_k_means_coresets_main/cmake-3.20.2/Tests/CMakeTests/downloads")

file(DOWNLOAD
  ${url}
  ${dir}/file3.png
  TIMEOUT 2
  STATUS status
  EXPECTED_HASH SHA1=5555555555555555555555555555555555555555
  )
