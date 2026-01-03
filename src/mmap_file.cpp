#include "nvdb/mmap_file.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <stdexcept>
#include <utility>
#include <cerrno>
#include <cstring>

namespace nvdb {

static std::runtime_error sys_error(const char* what) {
  return std::runtime_error(std::string(what) + ": " + std::strerror(errno));
}

// Implement destructor
MmapFile::~MmapFile() { close(); }

// Implement  move constructor
MmapFile::MmapFile(MmapFile&& other) noexcept { *this = std::move(other); }

//Implement  move assignment operator
MmapFile& MmapFile::operator=(MmapFile&& other) noexcept {
  if (this != &other) {
    close();
    fd_ = other.fd_;
    data_ = other.data_;
    size_ = other.size_;
    other.fd_ = -1;
    other.data_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

void MmapFile::open_readonly(const std::string& path) {
  close();

  fd_ = ::open(path.c_str(), O_RDONLY);  //using global ::open() from fcntl.h
  if (fd_ < 0) throw sys_error("open");

  // Check file size
  struct stat st {};
  if (::fstat(fd_, &st) != 0) throw sys_error("fstat");
  if (st.st_size <= 0) throw std::runtime_error("File size is zero");

  //Construct mmap
  size_ = static_cast<size_t>(st.st_size);
  void* p = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (p == MAP_FAILED) throw sys_error("mmap");

  data_ = static_cast<uint8_t*>(p);
}

// munmap()  & close file descriptor
void MmapFile::close() {
  if (data_) {
    ::munmap(data_, size_);
    data_ = nullptr;
    size_ = 0;
  }
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

} // namespace nvdb
