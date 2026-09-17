
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#define CUDA_KERNEL_ASSERT(x) ((void)0)
inline bool valid_spatial_level(int64_t height, int64_t width,
                                           int64_t start, int spatial_size) {
  // Check raw int64 metadata before narrowing. Subtraction after conversion to
  // unsigned handles zero, negative values and INT64_MIN without signed overflow.
  // The positive dimensions require start < spatial_size. Once bounded, use a
  // 32x32 -> 64-bit area product, including for malformed levels whose area does
  // not fit in int32; CUDA can use a wide 32-bit multiply here.
  const uint64_t limit = static_cast<uint64_t>(spatial_size);
  const bool valid =
      static_cast<uint64_t>(height) - 1 < limit &&
      static_cast<uint64_t>(width) - 1 < limit &&
      static_cast<uint64_t>(start) < limit &&
      static_cast<uint64_t>(static_cast<uint32_t>(height)) *
              static_cast<uint32_t>(width) <=
          limit - static_cast<uint64_t>(start);
  // PyTorch 2.4 has CUDA_KERNEL_ASSERT but not CUDA_KERNEL_ASSERT_MSG.
  CUDA_KERNEL_ASSERT(valid && "Spatial level exceeds the value tensor");
  return valid;
}
bool old(int64_t h,int64_t w,int64_t s,int n) {
 return h>0 && h<=n && w>0 && w<=n && s>=0 && s<=n && h*w<=n-s;
}
int main(){
 const int64_t lo=std::numeric_limits<int64_t>::min(), hi=std::numeric_limits<int64_t>::max();
 uint64_t count=0;
 for (int n : {1,2,46340,46341,1073741823,2147483647}) {
  std::vector<int64_t> a={lo,lo+1,-1,0,1,2,46340,46341,int64_t(n)-1,n,int64_t(n)+1,2147483647LL,2147483648LL,4294967297LL,hi};
  for(auto h:a) for(auto w:a) for(auto s:a){
   if(old(h,w,s,n)!=valid_spatial_level(h,w,s,n)) return 1;
   ++count;
  }
 }
 std::mt19937_64 gen(29);
 for(int i=0;i<100000;++i){
  int n=1+gen()%2147483647;
  int64_t h=static_cast<int64_t>(gen()),w=static_cast<int64_t>(gen()),s=static_cast<int64_t>(gen());
  if(old(h,w,s,n)!=valid_spatial_level(h,w,s,n)) return 2;
  ++count;
 }
 std::cout << "Actual helper equivalent to original: " << count << " cases; UBSan active; assertion-disabled bool-return checked\n";
}
