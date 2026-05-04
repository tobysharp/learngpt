#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

class ThreadPool {
 public:
  ThreadPool() : ThreadPool(std::thread::hardware_concurrency()) {}
  ThreadPool(int size);
  ~ThreadPool();

  template <typename Fn>
  void ParallelFor(int begin, int end, Fn&& fn) {
    const auto chunk = [&fn](int a, int b) {
      for (int i = a; i < b; ++i) fn(i);
    };

    auto item = std::make_shared<Item>(begin, end, begin, chunk, work_);
    while (!work_.compare_exchange_weak(item->next, item));
    cv_.notify_all();

    DoWork(item.get());

    while (int r = item->remaining)
      item->remaining.wait(r);
  }

 private:
  struct Item {
    const int begin, end;
    std::atomic<int> current;
    std::function<void(int, int)> chunk;
    std::shared_ptr<Item> next;
    std::atomic<int> remaining{end - begin};
  };
  void RunWorker();
  void DoWork(Item* item);

  std::vector<std::thread> threads_;
  std::atomic<bool> abort_{false};
  std::atomic<std::shared_ptr<Item>> work_{nullptr};
  mutable std::mutex mutex_;
  mutable std::condition_variable cv_;
};

inline ThreadPool::ThreadPool(int size) {
  threads_.reserve(size);
  for (int i = 0; i < size; ++i)
    threads_.emplace_back([this] { RunWorker(); });
}

inline ThreadPool::~ThreadPool() {
  abort_ = true;
  cv_.notify_all();
  for (auto& t : threads_) t.join();
}

inline void ThreadPool::RunWorker() {
  while (!abort_) {
    if (std::shared_ptr<Item> item = work_) {
      // Process the item if it has work remaining.
      if (item->current < item->end)
        DoWork(item.get());
      // Pop the work item if necessary.
      work_.compare_exchange_weak(item, item->next);
    } else {
      // Wait for the next event.
      std::unique_lock lock(mutex_);
      cv_.wait(lock, [this] { return abort_ || work_.load(); });
    }    
  }
}

inline void ThreadPool::DoWork(Item* item) {
  const int chunk_size = std::max<int>(1, (item->end - item->begin) / (2 * std::ssize(threads_)));

  while (!abort_) {
    const int after = (item->current += chunk_size);  // One atomic addition to claim work.
    const int begin = after - chunk_size;
    if (begin >= item->end) break;
    const int end = std::min(after, item->end);
    item->chunk(begin, end);
    if ((item->remaining -= (end - begin)) == 0)  // One atomic subtraction to signal completion.
      item->remaining.notify_one();
  }
}

inline ThreadPool& GlobalThreadPool() {
  static ThreadPool pool;
  return pool;
}

template <typename Fn>
void ParallelFor(int begin, int end, Fn&& fn) {
  GlobalThreadPool().ParallelFor(begin, end, std::forward<Fn>(fn));
}