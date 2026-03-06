#include "utility.hpp"

size_t getAlignedSize(size_t size, size_t alignment)
{
    return alignment > 0 ? (size + alignment - 1) & ~(alignment - 1) : size;
}

std::string readFile(const char* path)
{
    std::ostringstream buf;
    std::ifstream input(path);
    if (!input.is_open()) {
        logger.LogError("Could not open file for reading: ", path);
        return "";
    }
    buf << input.rdbuf();
    return buf.str();
}

void writeFile(const char* path, const std::string& content)
{
    std::ofstream o{ path, std::ofstream::out | std::ofstream::trunc };
    if (!o.is_open()) {
        logger.LogError("Could not open file for writing: ", path);
        return;
    }
    o << content;
}
