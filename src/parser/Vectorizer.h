//
// Created by oleg on 15.07.24.
//

#ifndef DECISION_TREE_VECTORIZER_H
#define DECISION_TREE_VECTORIZER_H

#include "iostream"
#include "vector"
#include "functional"

namespace transform {
    class Vectorizer {
    public:
        explicit Vectorizer(u_long num_args);
        std::vector<float> Vectorize(const std::vector<std::string>& input, std::vector<std::function<float(const std::string &)>>& vector_processors) const;
    private:
        u_long num_args_;
    };
}



#endif //DECISION_TREE_VECTORIZER_H
