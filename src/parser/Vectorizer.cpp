//
// Created by oleg on 15.07.24.
//

#include "Vectorizer.h"
#include "tbb/tbb.h"

transform::Vectorizer::Vectorizer(u_long num_args): num_args_(num_args) {}

std::vector<float>
transform::Vectorizer::Vectorize(const std::vector<std::string> &input, std::vector<std::function<float(const std::string&)>>& vector_processors) const {
    std::vector<float> r(input.size());
    tbb::parallel_for(tbb::blocked_range<size_t>(0, input.size(), num_args_ / 3),  // from input[0] to input[-1],
                    [&r, &input, &vector_processors](const tbb::blocked_range<size_t>& partition){
        for(size_t i = partition.begin(); i < partition.end(); ++i) { // Use partition.begin() instead of 0
            r[i] = (vector_processors[i])(input[i]);
        }
    });
    return r;
}

