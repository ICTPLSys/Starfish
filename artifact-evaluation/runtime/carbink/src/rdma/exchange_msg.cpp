#include "rdma/exchange_msg.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/ip.h>

#include <cstdlib>

#include "utils/debug.hpp"

namespace FarLib {

namespace tcp {

sockaddr_in get_sock_addr(const char *server_addr, const char *server_port) {
    in_addr_t addr = inet_addr(server_addr);
    ASSERT(addr != -1);
    in_port_t port = atoi(server_port);
    ASSERT(port != 0);
    sockaddr_in sock_addr = {
        .sin_family = AF_INET, .sin_port = htons(port), .sin_addr = {.s_addr = addr}};
    return sock_addr;
}

int listen_for_client(const char *server_addr, const char *server_port,
                      int &sock_fd) {
    sockaddr_in sock_addr = get_sock_addr(server_addr, server_port);
    sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd < 0) {
        ASSERT(sock_fd >= 0);
    }

    int on = 1;
    CHECK_ERR(setsockopt(sock_fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)));

    if (bind(sock_fd, reinterpret_cast<sockaddr *>(&sock_addr),
                   sizeof(sock_addr)) < 0) {
        return -1;
    }

    CHECK_ERR(listen(sock_fd, 1));

    sockaddr_in client_addr;
    socklen_t client_addr_len = sizeof(client_addr);
    int conn_fd = accept(sock_fd, reinterpret_cast<sockaddr *>(&client_addr),
                         &client_addr_len);
    if (conn_fd < 0) {
        return -1;
    }
    ASSERT(conn_fd >= 0);
    ASSERT(client_addr_len == sizeof(client_addr));
    return conn_fd;
}

int connect_to_server(const char *server_addr, const char *server_port) {
    sockaddr_in sock_addr = get_sock_addr(server_addr, server_port);
    int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd < 0) {
        return -1;
    }
    
    // Use blocking connect - let caller handle retries
    int ret = connect(sock_fd, reinterpret_cast<sockaddr *>(&sock_addr),
                      sizeof(sock_addr));
    
    if (ret < 0) {
        close(sock_fd);
        return -1;
    }
    
    return sock_fd;
}

}  // namespace tcp

}  // namespace FarLib
