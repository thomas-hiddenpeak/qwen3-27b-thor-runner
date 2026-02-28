// KV Cache Offload — Disk 后端实现
#include "disk_backend.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdio>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>

namespace qwen_thor {
namespace cache {

// 文件头 magic 和版本
static constexpr uint64_t DISK_MAGIC   = 0x4B56434F66666C64ULL;  // "KVCOffld"
static constexpr uint32_t DISK_VERSION = 1;

// 递归创建目录 (等价于 mkdir -p)
static void mkdir_p(const std::string& path) {
    size_t pos = 0;
    while ((pos = path.find('/', pos + 1)) != std::string::npos) {
        mkdir(path.substr(0, pos).c_str(), 0755);
    }
    mkdir(path.c_str(), 0755);
}

DiskBackend::DiskBackend(const std::string& cache_dir, size_t max_bytes)
    : cache_dir_(cache_dir), max_bytes_(max_bytes) {
    // 递归创建缓存目录
    mkdir_p(cache_dir);
    std::cout << "[DiskBackend] Initialized at " << cache_dir
              << " with max " << (max_bytes / (1024*1024)) << " MB" << std::endl;
}

DiskBackend::~DiskBackend() {
    // 不删除磁盘文件 (持久缓存)
}

std::string DiskBackend::key_to_path(const CacheKey& key) const {
    std::ostringstream oss;
    oss << cache_dir_ << "/" << std::hex << std::setfill('0')
        << std::setw(16) << key.hash << ".bin";
    return oss.str();
}

bool DiskBackend::contains(const CacheKey& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return index_.find(key) != index_.end();
}

int DiskBackend::prefix_match(const std::vector<CacheKey>& keys) const {
    std::lock_guard<std::mutex> lock(mutex_);
    int matched = 0;
    for (const auto& key : keys) {
        if (index_.find(key) == index_.end()) break;
        matched++;
    }
    return matched;
}

bool DiskBackend::put(const CacheKey& key, CacheEntryPtr entry) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (index_.find(key) != index_.end()) return false;

    size_t file_bytes = entry->total_bytes() + 256;  // 头部开销预估
    if (file_bytes > max_bytes_) return false;

    evict_until_fit(file_bytes);

    std::string path = key_to_path(key);
    if (!write_entry(path, *entry)) return false;

    lru_list_.emplace_front(key, FileInfo{file_bytes});
    index_[key] = lru_list_.begin();
    current_bytes_ += file_bytes;

    return true;
}

CacheEntryPtr DiskBackend::get(const CacheKey& key) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = index_.find(key);
    if (it == index_.end()) return nullptr;

    // LRU 更新
    auto list_it = it->second;
    if (list_it != lru_list_.begin()) {
        lru_list_.splice(lru_list_.begin(), lru_list_, list_it);
    }

    // 从磁盘读取
    std::string path = key_to_path(key);
    return read_entry(path);
}

bool DiskBackend::remove(const CacheKey& key) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = index_.find(key);
    if (it == index_.end()) return false;

    auto list_it = it->second;
    current_bytes_ -= list_it->second.file_bytes;

    // 删除磁盘文件
    std::string path = key_to_path(key);
    std::remove(path.c_str());

    lru_list_.erase(list_it);
    index_.erase(it);
    return true;
}

size_t DiskBackend::current_size_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_bytes_;
}

size_t DiskBackend::max_size_bytes() const { return max_bytes_; }

size_t DiskBackend::num_entries() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return index_.size();
}

std::vector<CacheKey> DiskBackend::get_evict_candidates(int n) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<CacheKey> candidates;
    int count = 0;
    for (auto it = lru_list_.rbegin(); it != lru_list_.rend() && count < n; ++it, ++count) {
        candidates.push_back(it->first);
    }
    return candidates;
}

void DiskBackend::evict_until_fit(size_t needed_bytes) {
    while (current_bytes_ + needed_bytes > max_bytes_ && !lru_list_.empty()) {
        auto& back = lru_list_.back();
        std::string path = key_to_path(back.first);
        size_t evicted_bytes = back.second.file_bytes;
        std::remove(path.c_str());
        current_bytes_ -= evicted_bytes;
        index_.erase(back.first);
        lru_list_.pop_back();

        // 通知 Monitor
        if (eviction_cb_) {
            eviction_cb_(evicted_bytes);
        }
    }
}

// ---------------------------------------------------------------------------
// 序列化格式 (binary, little-endian):
//   [8]  magic
//   [4]  version
//   [4]  num_tokens
//   [8]  kv_bytes
//   [8]  ssm_bytes
//   [8]  conv_bytes
//   [num_tokens*4]  token_ids
//   [kv_bytes]      KV data
//   [ssm_bytes]     SSM data
//   [conv_bytes]    Conv data
// ---------------------------------------------------------------------------
bool DiskBackend::write_entry(const std::string& path, const CacheEntry& entry) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        std::cerr << "[DiskBackend] Failed to open " << path << " for writing" << std::endl;
        return false;
    }

    uint64_t magic = DISK_MAGIC;
    uint32_t version = DISK_VERSION;
    int32_t  num_tokens = entry.num_tokens;
    uint64_t kv_bytes  = entry.kv_data.size();
    uint64_t ssm_bytes = entry.ssm_data.size();
    uint64_t conv_bytes = entry.conv_data.size();

    ofs.write(reinterpret_cast<const char*>(&magic),      8);
    ofs.write(reinterpret_cast<const char*>(&version),    4);
    ofs.write(reinterpret_cast<const char*>(&num_tokens), 4);
    ofs.write(reinterpret_cast<const char*>(&kv_bytes),   8);
    ofs.write(reinterpret_cast<const char*>(&ssm_bytes),  8);
    ofs.write(reinterpret_cast<const char*>(&conv_bytes), 8);

    // Token IDs
    ofs.write(reinterpret_cast<const char*>(entry.tokens.data()),
              entry.tokens.size() * sizeof(int));

    // Data blobs
    if (kv_bytes > 0)
        ofs.write(reinterpret_cast<const char*>(entry.kv_data.data()), kv_bytes);
    if (ssm_bytes > 0)
        ofs.write(reinterpret_cast<const char*>(entry.ssm_data.data()), ssm_bytes);
    if (conv_bytes > 0)
        ofs.write(reinterpret_cast<const char*>(entry.conv_data.data()), conv_bytes);

    return ofs.good();
}

CacheEntryPtr DiskBackend::read_entry(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return nullptr;

    uint64_t magic;
    uint32_t version;
    int32_t  num_tokens;
    uint64_t kv_bytes, ssm_bytes, conv_bytes;

    ifs.read(reinterpret_cast<char*>(&magic),      8);
    ifs.read(reinterpret_cast<char*>(&version),    4);
    ifs.read(reinterpret_cast<char*>(&num_tokens), 4);
    ifs.read(reinterpret_cast<char*>(&kv_bytes),   8);
    ifs.read(reinterpret_cast<char*>(&ssm_bytes),  8);
    ifs.read(reinterpret_cast<char*>(&conv_bytes), 8);

    if (magic != DISK_MAGIC || version != DISK_VERSION) {
        std::cerr << "[DiskBackend] Invalid file: " << path << std::endl;
        return nullptr;
    }

    auto entry = std::make_shared<CacheEntry>();
    entry->num_tokens = num_tokens;

    // Token IDs
    entry->tokens.resize(num_tokens);
    ifs.read(reinterpret_cast<char*>(entry->tokens.data()), num_tokens * sizeof(int));

    // Data blobs
    entry->kv_data.resize(kv_bytes);
    if (kv_bytes > 0)
        ifs.read(reinterpret_cast<char*>(entry->kv_data.data()), kv_bytes);

    entry->ssm_data.resize(ssm_bytes);
    if (ssm_bytes > 0)
        ifs.read(reinterpret_cast<char*>(entry->ssm_data.data()), ssm_bytes);

    entry->conv_data.resize(conv_bytes);
    if (conv_bytes > 0)
        ifs.read(reinterpret_cast<char*>(entry->conv_data.data()), conv_bytes);

    if (!ifs) {
        std::cerr << "[DiskBackend] Read error: " << path << std::endl;
        return nullptr;
    }

    return entry;
}

} // namespace cache
} // namespace qwen_thor
