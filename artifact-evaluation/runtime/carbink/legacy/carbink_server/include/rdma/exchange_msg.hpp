/**
 * Exchange message by TCP to build RDMA connection
 */
#pragma once

#include <sys/socket.h>
#include <unistd.h>

#include <cstddef>

#include "utils/debug.hpp"
namespace FarLib {

namespace tcp {

// open an server and listen for a msg, return: conn_fd
int listen_for_client(const char *server_addr, const char *server_port,
                      int &sock_fd);

int connect_to_server(const char *server_addr, const char *server_port);

inline ssize_t recieve_all(int fd, void *buf, size_t n, int flags) {
    size_t recieved = 0;
    while (recieved < n) {
        ssize_t r = recv(fd, buf, n - recieved, flags);
        ASSERT(r >= 0);
        recieved += static_cast<size_t>(r);
        ASSERT(recieved <= n);
        buf = static_cast<void *>(static_cast<std::byte *>(buf) + r);
    }
    return static_cast<ssize_t>(recieved);
}

inline ssize_t send_all(int fd, const void *buf, size_t n, int flags) {
    size_t sent = 0;
    while (sent < n) {
        ssize_t s = send(fd, static_cast<const std::byte *>(buf) + sent, n - sent, flags);
        ASSERT(s >= 0);
        sent += static_cast<size_t>(s);
        ASSERT(sent <= n);
    }
    return static_cast<ssize_t>(sent);
}

}  // namespace tcp

}  // namespace FarLib