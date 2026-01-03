#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace nvdb {

class MmapFile {
public:
  MmapFile() = default;
  ~MmapFile();

  // Non-copyable
  MmapFile(const MmapFile&) = delete;
  MmapFile& operator=(const MmapFile&) = delete;

  // Movable
  MmapFile(MmapFile&& other) noexcept;
  MmapFile& operator=(MmapFile&& other) noexcept;

  void open_readonly(const std::string& path);

  const uint8_t* data() const { return data_; }
  size_t size() const { return size_; }
  bool is_open() const { return data_ != nullptr; }

private:
  int fd_ = -1;  // File Descriptor
  uint8_t* data_ = nullptr; // data ptr
  size_t size_ = 0; // file size

  void close();
};

} // namespace nvdb
