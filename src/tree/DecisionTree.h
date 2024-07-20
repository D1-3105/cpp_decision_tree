//
// Created by oleg on 15.07.24.
//

#ifndef DECISION_TREE_DECISION_TREE_H
#define DECISION_TREE_DECISION_TREE_H

#include "cmath"
#include "iostream"
#include <memory>
#include <vector>

namespace ai {
    using data_set = std::vector<std::vector<float>>;
    struct Node {
        Node* pos_child;
        Node* neg_child;
        float bound;
        size_t attr_idx;
        float result;

        Node* choice(std::vector<float> values) {
            if (values[attr_idx] <= bound) {
                return neg_child;
            }
            return pos_child;
        }

        bool is_result() {
            if (pos_child == nullptr and neg_child == nullptr) {
                return true;
            }
            return false;
        }
    };

    class DecisionTree {
    public:
        DecisionTree(std::vector<std::vector<float>>& T, size_t result_idx);
        float Predict(std::vector<float>&);
    private:
        Node* genesis_;
    protected:
        void BuildBranch(data_set& current_dataset, size_t result_idx, Node*& branch_tail, bool link_pos);
    };
}



#endif //DECISION_TREE_DECISION_TREE_H
