// Keep the transaction coordinator out of the large application/cache header.
// It runs once per update batch; applications do not need to instantiate it.
#include "cache/cache.hpp"
#include "cache/core/rdma/ec_update_path.ipp"
#include "cache/core/rdma/ec_rmw_path.ipp"
