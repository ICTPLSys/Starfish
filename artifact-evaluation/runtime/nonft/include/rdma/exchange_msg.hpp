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
    ssize_t recieved = 0;
    while (recieved < n) {
        ssize_t r = recv(fd, buf, n - recieved, flags);
        ASSERT(r > 0);
        recieved += r;
        ASSERT(recieved <= n);
        buf = static_cast<void *>(static_cast<std::byte *>(buf) + r);
    }
    return recieved;
}

inline ssize_t send_all(int fd, const void *buf, size_t n, int flags) {
    ssize_t sent = 0;
    while (sent < n) {
        ssize_t r = send(fd, buf, n - sent, flags);
        ASSERT(r > 0);
        sent += r;
        ASSERT(sent <= n);
        buf = static_cast<const void *>(static_cast<const std::byte *>(buf) + r);
    }
    return sent;
}

}  // namespace tcp

}  // namespace FarLib
