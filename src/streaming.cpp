#include "vtlc/streaming.hpp"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <future>
#include <limits>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
namespace vtlc {
namespace {
void put(std::vector<uint8_t> &b, uint64_t x, unsigned n) {
  for (unsigned i = 0; i < n; ++i)
    b.push_back(uint8_t(x >> (i * 8)));
}
struct Reader {
  std::span<const uint8_t> b;
  size_t pos = 0;
  uint64_t get(unsigned n) {
    if (n > b.size() - pos)
      throw std::runtime_error("truncated stream");
    uint64_t x = 0;
    for (unsigned i = 0; i < n; ++i)
      x |= uint64_t(b[pos++]) << (i * 8);
    return x;
  }
  std::span<const uint8_t> take(size_t n) {
    if (n > b.size() - pos)
      throw std::runtime_error("truncated chunk");
    auto x = b.subspan(pos, n);
    pos += n;
    return x;
  }
};
uint64_t product(std::span<const uint64_t> s) {
  uint64_t n = 1;
  for (auto d : s) {
    if (d && n > UINT64_MAX / d)
      throw std::runtime_error("shape overflow");
    n *= d;
  }
  return n;
}
class WorkerPool {
  std::mutex mutex_;
  std::condition_variable wake_;
  std::queue<std::function<void()>> tasks_;
  std::vector<std::thread> workers_;
  bool stopped_ = false;

public:
  WorkerPool() {
    unsigned n =
        std::min(32u, std::max(1u, std::thread::hardware_concurrency()));
    try {
      for (unsigned i = 0; i < n; i++)
        workers_.emplace_back([this] {
          for (;;) {
            std::function<void()> task;
            {
              std::unique_lock lock(mutex_);
              wake_.wait(lock, [&] { return stopped_ || !tasks_.empty(); });
              if (stopped_ && tasks_.empty())
                return;
              task = std::move(tasks_.front());
              tasks_.pop();
            }
            task();
          }
        });
    } catch (...) {
      {
        std::lock_guard lock(mutex_);
        stopped_ = true;
      }
      wake_.notify_all();
      for (auto &t : workers_)
        t.join();
      throw;
    }
  }
  ~WorkerPool() {
    {
      std::lock_guard lock(mutex_);
      stopped_ = true;
    }
    wake_.notify_all();
    for (auto &t : workers_)
      t.join();
  }
  std::future<void> submit(std::function<void()> work) {
    auto task = std::make_shared<std::packaged_task<void()>>(std::move(work));
    auto result = task->get_future();
    {
      std::lock_guard lock(mutex_);
      tasks_.push([task] { (*task)(); });
    }
    wake_.notify_one();
    return result;
  }
};
WorkerPool &workers() {
  static WorkerPool pool;
  return pool;
}
} // namespace
struct StreamEncoder::Impl {
  std::vector<uint64_t> shape;
  Options opt;
  uint64_t elements = 0, frames = 0, pending_frames = 0;
  bool done = false, drained = false;
  std::vector<float> pending;
  std::vector<std::vector<uint8_t>> chunks;
  void flush() {
    if (!pending_frames)
      return;
    std::vector<uint64_t> s{pending_frames};
    s.insert(s.end(), shape.begin(), shape.end());
    auto o = opt;
    o.time_axis = 0;
    chunks.push_back(compress(pending, s, o));
    pending.clear();
    pending_frames = 0;
  }
};
StreamEncoder::StreamEncoder(std::span<const uint64_t> s, Options o)
    : impl_(std::make_unique<Impl>()) {
  if (s.size() > 31 || !o.keyframe_interval || o.time_axis != 0)
    throw std::runtime_error("invalid streaming options");
  impl_->shape.assign(s.begin(), s.end());
  impl_->opt = o;
  impl_->elements = product(s);
  if (impl_->elements > o.memory_limit / 4)
    throw std::runtime_error("frame exceeds memory limit");
}
StreamEncoder::~StreamEncoder() = default;
StreamEncoder::StreamEncoder(StreamEncoder &&) noexcept = default;
StreamEncoder &StreamEncoder::operator=(StreamEncoder &&) noexcept = default;
void StreamEncoder::encode_frame(std::span<const float> frame) {
  auto &s = *impl_;
  if (s.done || frame.size() != s.elements)
    throw std::runtime_error("invalid streaming frame");
  if (s.frames == UINT64_MAX)
    throw std::runtime_error("too many frames");
  if (s.elements &&
      (s.pending_frames + 1) > s.opt.memory_limit / 4 / s.elements)
    throw std::runtime_error("chunk exceeds memory limit");
  size_t old = s.pending.size();
  if (frame.size() > s.pending.max_size() - old)
    throw std::runtime_error("streaming chunk is too large");
  s.pending.resize(old + frame.size());
  if (!frame.empty())
    std::memcpy(s.pending.data() + old, frame.data(), frame.size_bytes());
  ++s.frames;
  ++s.pending_frames;
  if (s.pending_frames == s.opt.keyframe_interval)
    s.flush();
}
std::optional<std::vector<uint8_t>> StreamEncoder::take_chunk() {
  auto &s = *impl_;
  if (s.done)
    throw std::runtime_error("stream already finished");
  if (s.chunks.empty())
    return std::nullopt;
  s.drained = true;
  auto result = std::move(s.chunks.front());
  s.chunks.erase(s.chunks.begin());
  return result;
}
std::vector<std::vector<uint8_t>> StreamEncoder::finish_chunks() {
  auto &s = *impl_;
  if (s.done)
    throw std::runtime_error("stream already finished");
  s.flush();
  s.done = true;
  return std::move(s.chunks);
}
std::vector<uint8_t> StreamEncoder::finish() {
  auto &s = *impl_;
  if (s.done || s.drained)
    throw std::runtime_error("stream finished or chunks already drained");
  s.flush();
  s.done = true;
  std::vector<uint8_t> out;
  put(out, 0x31535456, 4);
  put(out, 1, 4);
  put(out, s.shape.size(), 4);
  put(out, 0, 4);
  put(out, s.frames, 8);
  put(out, s.chunks.size(), 8);
  for (auto x : s.shape)
    put(out, x, 8);
  for (auto &c : s.chunks) {
    put(out, c.size(), 8);
    out.insert(out.end(), c.begin(), c.end());
  }
  return out;
}
struct StreamDecoder::Impl {
  std::vector<uint64_t> shape;
  std::vector<std::span<const uint8_t>> chunks;
  Backend backend;
  uint64_t limit, frames = 0, elements = 0, next = 0;
  size_t chunk = 0;
  uint64_t within = 0;
  Tensor current;
};
StreamDecoder::StreamDecoder(std::span<const uint8_t> data, Backend be,
                             uint64_t limit)
    : impl_(std::make_unique<Impl>()) {
  auto &s = *impl_;
  s.backend = be;
  s.limit = limit;
  if (!available(be))
    throw std::runtime_error("backend unavailable");
  Reader r{data};
  if (r.get(4) != 0x31535456 || r.get(4) != 1)
    throw std::runtime_error("invalid streaming format");
  uint64_t rank = r.get(4);
  if (rank > 31 || r.get(4))
    throw std::runtime_error("invalid streaming rank");
  s.frames = r.get(8);
  uint64_t count = r.get(8);
  if (count > (data.size() - r.pos) / 8)
    throw std::runtime_error("invalid chunk count");
  for (uint64_t i = 0; i < rank; i++)
    s.shape.push_back(r.get(8));
  s.elements = product(s.shape);
  if (s.elements > limit / 4)
    throw std::runtime_error("frame exceeds memory limit");
  uint64_t sum = 0;
  for (uint64_t i = 0; i < count; i++) {
    uint64_t n = r.get(8);
    if (n > data.size() - r.pos || n > SIZE_MAX)
      throw std::runtime_error("invalid chunk size");
    auto c = r.take(size_t(n));
    auto m = inspect(c, limit);
    if (m.time_axis != 0 || m.shape.size() != rank + 1 || !m.shape[0])
      throw std::runtime_error("invalid chunk shape");
    for (size_t j = 0; j < rank; j++)
      if (m.shape[j + 1] != s.shape[j])
        throw std::runtime_error("frame shape mismatch");
    if (m.shape[0] > s.frames - sum)
      throw std::runtime_error("invalid frame count");
    sum += m.shape[0];
    s.chunks.push_back(c);
  }
  if (sum != s.frames || r.pos != data.size())
    throw std::runtime_error("invalid stream length");
}
StreamDecoder::~StreamDecoder() = default;
StreamDecoder::StreamDecoder(StreamDecoder &&) noexcept = default;
StreamDecoder &StreamDecoder::operator=(StreamDecoder &&) noexcept = default;
size_t StreamDecoder::chunk_count() const { return impl_->chunks.size(); }
uint64_t StreamDecoder::frame_count() const { return impl_->frames; }
std::span<const uint64_t> StreamDecoder::frame_shape() const {
  return impl_->shape;
}
Tensor StreamDecoder::decode_chunk(size_t i) const {
  if (i >= impl_->chunks.size())
    throw std::out_of_range("chunk index");
  return decompress(impl_->chunks[i], impl_->backend, impl_->limit);
}
Tensor StreamDecoder::decode_all_parallel(unsigned threads) const {
  const auto &s = *impl_;
  if (s.elements && s.frames > s.limit / 4 / s.elements)
    throw std::runtime_error("tensor exceeds memory limit");
  uint64_t total = s.frames * s.elements;
  if (total > SIZE_MAX / 4)
    throw std::runtime_error("tensor too large");
  Tensor out;
  out.shape.push_back(s.frames);
  out.shape.insert(out.shape.end(), s.shape.begin(), s.shape.end());
  out.bits.resize(size_t(total));
  if (s.chunks.empty())
    return out;
  std::vector<size_t> offsets(s.chunks.size());
  size_t pos = 0;
  for (size_t i = 0; i < s.chunks.size(); i++) {
    offsets[i] = pos;
    auto m = inspect(s.chunks[i], s.limit);
    pos += size_t(m.elements);
  }
  if (!threads) {
    threads = std::max(1u, std::thread::hardware_concurrency());
  }
  threads = unsigned(std::min<size_t>({threads, s.chunks.size(), 33}));
  std::atomic_size_t next{0};
  std::atomic_bool failed{false};
  std::exception_ptr error;
  std::mutex mutex;
  auto worker = [&] {
    try {
      while (!failed) {
        size_t i = next.fetch_add(1);
        if (i >= s.chunks.size())
          break;
        auto chunk = decode_chunk(i);
        if (!chunk.bits.empty())
          std::memcpy(out.bits.data() + offsets[i], chunk.bits.data(),
                      chunk.bits.size() * 4);
      }
    } catch (...) {
      failed = true;
      std::lock_guard lock(mutex);
      if (!error)
        error = std::current_exception();
    }
  };
  std::vector<std::future<void>> jobs;
  jobs.reserve(threads - 1);
  try {
    for (unsigned i = 1; i < threads; i++)
      jobs.push_back(workers().submit(worker));
    worker();
  } catch (...) {
    failed = true;
    for (auto &job : jobs)
      job.wait();
    throw;
  }
  for (auto &job : jobs)
    job.get();
  if (error)
    std::rethrow_exception(error);
  return out;
}
bool StreamDecoder::decode_next(std::vector<uint32_t> &frame) {
  auto &s = *impl_;
  if (s.next == s.frames)
    return false;
  if (s.current.shape.empty() || s.within == s.current.shape[0]) {
    s.current = decode_chunk(s.chunk++);
    s.within = 0;
  }
  frame.resize(s.elements);
  if (s.elements)
    std::memcpy(frame.data(), s.current.bits.data() + s.within * s.elements,
                s.elements * 4);
  ++s.within;
  ++s.next;
  return true;
}
} // namespace vtlc
