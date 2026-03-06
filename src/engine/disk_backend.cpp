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
#include <fcntl.h>  // posix_fadvise

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

// 释放文件对应的 page cache, 防止 SSD 缓存操作吃掉物理内存
// Thor 128 GB 统一内存, page cache 占用会挤压 GPU 可用内存
static void drop_page_cache(const std::string& path) {
    int fd = open(path.c_str(), O_RDONLY);
    if (fd >= 0) {
        posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        close(fd);
    }
}

// 读取 .bin 文件头部元数据 (不加载完整数据)
// 返回: {file_bytes, has_ssm, hash, valid}
struct FileHeaderInfo {
    size_t file_bytes;
    bool has_ssm;
    uint64_t hash;
    bool valid;
};

static FileHeaderInfo read_file_header(const std::string& path) {
    FileHeaderInfo info{0, false, 0, false};

    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) return info;

    info.file_bytes = ifs.tellg();
    ifs.seekg(0);

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

    if (!ifs || magic != DISK_MAGIC || version != DISK_VERSION) return info;

    info.has_ssm = (ssm_bytes > 0);
    info.valid = true;
    return info;
}

DiskBackend::DiskBackend(const std::string& cache_dir, size_t max_bytes)
    : cache_dir_(cache_dir), max_bytes_(max_bytes) {
    // 递归创建缓存目录
    mkdir_p(cache_dir);

    // 扫描已有缓存文件, 重建索引 (持久缓存启动恢复)
    scan_existing_files();

    std::cout << "[DiskBackend] Initialized at " << cache_dir
              << " with max " << (max_bytes / (1024*1024)) << " MB"
              << ", recovered " << index_.size() << " entries ("
              << (current_bytes_ / (1024*1024)) << " MB)" << std::endl;
}

DiskBackend::~DiskBackend() {
    // 不删除磁盘文件 (持久缓存)
}

void DiskBackend::scan_existing_files() {
    DIR* dir = opendir(cache_dir_.c_str());
    if (!dir) return;

    struct dirent* ent;
    int recovered = 0, invalid = 0;

    while ((ent = readdir(dir)) != nullptr) {
        std::string name = ent->d_name;
        // 只处理 .bin 文件
        if (name.size() < 5 || name.substr(name.size() - 4) != ".bin") continue;

        // 从文件名解析 hash (16 位 hex)
        std::string hex_part = name.substr(0, name.size() - 4);
        uint64_t hash;
        try {
            hash = std::stoull(hex_part, nullptr, 16);
        } catch (...) {
            continue;  // 文件名格式不正确, 跳过
        }

        CacheKey key{hash};
        if (index_.find(key) != index_.end()) continue;  // 已存在

        std::string path = cache_dir_ + "/" + name;
        auto header = read_file_header(path);
        if (!header.valid) {
            // 无效文件, 删除
            std::remove(path.c_str());
            invalid++;
            continue;
        }

        // 检查容量限制
        if (current_bytes_ + header.file_bytes > max_bytes_) {
            // 超出容量, 删除旧文件
            std::remove(path.c_str());
            continue;
        }

        lru_list_.emplace_back(key, FileInfo{header.file_bytes, header.has_ssm});
        auto it = std::prev(lru_list_.end());
        index_[key] = it;
        current_bytes_ += header.file_bytes;
        recovered++;
    }

    closedir(dir);

    if (invalid > 0) {
        std::cerr << "[DiskBackend] Removed " << invalid << " invalid cache files" << std::endl;
    }
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
    drop_page_cache(path);  // 释放 page cache

    lru_list_.emplace_front(key, FileInfo{file_bytes, entry->has_ssm_state()});
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
    auto entry = read_entry(path);
    drop_page_cache(path);  // 释放 page cache
    return entry;
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

bool DiskBackend::has_ssm_state(const CacheKey& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = index_.find(key);
    if (it == index_.end()) return false;
    return it->second->second.has_ssm;
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
