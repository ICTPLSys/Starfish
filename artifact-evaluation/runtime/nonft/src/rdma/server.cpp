#include "rdma/server.hpp"

#include <iostream>
#include <cstdlib>

using namespace FarLib::rdma;

int main(int argc, const char *const argv[]) {
    if (argc != 2) {
        std::cerr << argv[0] << " <configure file>" << std::endl;
        std::abort();
    }
    const char *configure_file_name = argv[1];
    Configure config;
    config.from_file(configure_file_name);
    
    // Allow overriding server_port via environment variable or command line
    // Priority: SERVER_PORT env > argv[2] > config file
    const char* env_port = std::getenv("SERVER_PORT");
    if (env_port != nullptr) {
        config.server_port = env_port;
        config.post_process_config();
    } else if (argc >= 3) {
        config.server_port = argv[2];
        config.post_process_config();
    }
    
    std::cout << "Starting server on port " << config.server_port << std::endl;
    
    Server server(config);
    while (true) {
        server.start();
    }
    std::cout << "Bye!" << std::endl;
    return 0;
}