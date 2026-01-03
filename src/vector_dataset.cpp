#include "nvdb/vector_dataset.h"

#include <stdexcept>
#include <cstddef>
#include <cstdint>

namespace nvdb {

// Fallback raw12 header (always float32)
#pragma pack(push, 1)
struct Raw12Header {
  uint32_t count;
  uint32_t reserved;
  uint32_t dim;
};
#pragma pack(pop)

static bool size_matches_raw12(size_t file_size, uint32_t count, uint32_t dim) {
  const size_t need = sizeof(Raw12Header)
    + static_cast<size_t>(count) * static_cast<size_t>(dim) * sizeof(float);
  return file_size == need;
}

void VectorDataset::load(const std::string& path) {
  mm_.open_readonly(path);

  // Reset state
  count_ = 0;
  dim_ = 0;
  dtype_ = static_cast<uint32_t>(DType::Float32);
  data_offset_ = 0;
  vectors_f32_ = nullptr;
  vectors_f16_ = nullptr;

  // --- Try vecbin64 format (64B header) ---
  if (mm_.size() >= sizeof(VecbinHeader)) {
    const auto* h = reinterpret_cast<const VecbinHeader*>(mm_.data());

    const bool header_ok =
      (h->magic == kMagic) &&
      (h->version == kVersion) &&
      (h->dim > 0) &&
      (h->count > 0) &&
      (h->dtype == static_cast<uint32_t>(DType::Float32) ||
       h->dtype == static_cast<uint32_t>(DType::Float16));

    if (header_ok) {
      dim_ = h->dim;
      count_ = h->count;
      dtype_ = h->dtype;
      data_offset_ = sizeof(VecbinHeader);

      const size_t expect = sizeof(VecbinHeader) + bytes_for_vectors_typed(count_, dim_, dtype_);
      if (mm_.size() != expect) {
        throw std::runtime_error("VecbinHeader ok but file size mismatch");
      }

      const uint8_t* payload = mm_.data() + data_offset_;
      if (dtype_ == static_cast<uint32_t>(DType::Float32)) {
        vectors_f32_ = reinterpret_cast<const float*>(payload);
        vectors_f16_ = nullptr;
      } else { // Float16
        vectors_f16_ = reinterpret_cast<const uint16_t*>(payload);
        vectors_f32_ = nullptr;
      }
      return;
    }
  }

  // --- Fallback raw12: [u32 count][u32 reserved][u32 dim] + float32 payload ---
  if (mm_.size() < sizeof(Raw12Header)) {
    throw std::runtime_error("File too small (neither vecbin64 nor raw12)");
  }

  const auto* r = reinterpret_cast<const Raw12Header*>(mm_.data());
  if (r->dim == 0 || r->count == 0) {
    throw std::runtime_error("raw12 header invalid (count/dim == 0)");
  }
  if (!size_matches_raw12(mm_.size(), r->count, r->dim)) {
    throw std::runtime_error("raw12 header parsed but file size mismatch");
  }

  dim_ = r->dim;
  count_ = static_cast<uint64_t>(r->count);
  dtype_ = static_cast<uint32_t>(DType::Float32);
  data_offset_ = sizeof(Raw12Header);

  vectors_f32_ = reinterpret_cast<const float*>(mm_.data() + data_offset_);
  vectors_f16_ = nullptr;
}

const float* VectorDataset::vector_ptr_f32(uint64_t i) const {
  if (dtype_ != static_cast<uint32_t>(DType::Float32) || !vectors_f32_) {
    throw std::runtime_error("Dataset is not float32");
  }
  if (i >= count_) throw std::runtime_error("Index out of range");
  return vectors_f32_ + static_cast<size_t>(i) * static_cast<size_t>(dim_);
}

const uint16_t* VectorDataset::vector_ptr_f16(uint64_t i) const {
  if (dtype_ != static_cast<uint32_t>(DType::Float16) || !vectors_f16_) {
    throw std::runtime_error("Dataset is not float16");
  }
  if (i >= count_) throw std::runtime_error("Index out of range");
  return vectors_f16_ + static_cast<size_t>(i) * static_cast<size_t>(dim_);
}

} // namespace nvdb
