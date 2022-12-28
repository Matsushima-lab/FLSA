#include <vector>
#include <string>

template <typename T>
void print(std::vector<T> const &input);

std::vector<std::string> split(std::string str, char del);

std::vector<std::vector<double> >
csv2vector(std::string filename, int ignore_line_num);

template <typename T>
void print(std::vector<T> const &input, std::vector<int> const &index);

