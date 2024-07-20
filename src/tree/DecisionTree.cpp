#include "DecisionTree.h"
#include "map"
#include "tuple"
#include "stdexcept"

float ai::DecisionTree::Predict(std::vector<float>& values) {
    Node* current = genesis_;
    while (!current->is_result()) {
        current = current->choice(values);
    }
    return current->result;
}

std::tuple<size_t, float, ai::data_set, ai::data_set>
BuildMeanDecisionMap(size_t result_idx, size_t num_attrs, ai::data_set& set) {
    std::map<size_t, std::tuple<float, ai::data_set, ai::data_set>> each_attr;

    for (size_t attr_chosen = 0; attr_chosen < num_attrs; attr_chosen++) {
        if (result_idx == attr_chosen) {
            continue;
        }
        float sum = 0.0f;
        size_t count = 0;
        for (const auto& decision : set) {
            count++;
            sum += decision[attr_chosen];
        }
        if (count == 0) continue; // Avoid division by zero
        float bound = sum / float(count);
        ai::data_set pos_set, neg_set;
        for (const auto& decision : set) {
            if (decision[attr_chosen] < bound) {
                pos_set.push_back(decision);
            } else {
                neg_set.push_back(decision);
            }
        }
        if (!pos_set.empty() && !neg_set.empty()) {
            each_attr[attr_chosen] = {bound, pos_set, neg_set};
        }
    }

    if (each_attr.empty()) {
        throw std::logic_error("All possible splits resulted in empty sets!");
    }

    std::tuple<size_t, float, ai::data_set, ai::data_set> best_attr_split = {0, 0.0f, ai::data_set(), ai::data_set()};
    for (const auto& current_attr_pair : each_attr) {
        size_t attr = current_attr_pair.first;
        auto payload = current_attr_pair.second;
        ai::data_set pos_new = std::get<1>(payload);
        ai::data_set neg_new = std::get<2>(payload);
        ai::data_set pos = std::get<2>(best_attr_split);
        ai::data_set neg = std::get<3>(best_attr_split);
        if (pos.size() + neg.size() < pos_new.size() + neg_new.size()) {
            best_attr_split = {attr, std::get<0>(payload), pos_new, neg_new};
        }
    }

    std::cout << "Chosen attribute: " << std::get<0>(best_attr_split) << ", Bound: " << std::get<1>(best_attr_split) << "\n";
    std::cout << "Positive split size: " << std::get<2>(best_attr_split).size() << ", Negative split size: " << std::get<3>(best_attr_split).size() << "\n";

    return best_attr_split;
}


ai::Node* from_decision_info(
        std::tuple<size_t, float, ai::data_set, ai::data_set> info
) {
    using namespace std;
    return new ai::Node{nullptr, nullptr, get<1>(info), get<0>(info), 0.0};
}

std::tuple<bool, float> is_homogenous(ai::data_set& data, size_t result_idx) {
    if (data.empty()) {
        throw std::logic_error("dataset cannot be estimated as homogenous if empty!");
    }
    const auto& firstRow = data.front();
    float r = firstRow[result_idx];

    for (const auto& row : data) {
        if (row[result_idx] != r) {
            return {false, 0.0};
        }
    }
    return {true, r};
}

struct SplitInfo {
    ai::data_set pos_dataset;
    ai::data_set neg_dataset;
    size_t attr_idx{};
    float bound{};
    bool pos_homo{};
    bool neg_homo{};
    float pos_r{};
    float neg_r{};
};

SplitInfo collect_info(size_t result_idx, ai::data_set& current_dataset) {
    SplitInfo info;
    std::tie(info.attr_idx, info.bound, info.pos_dataset, info.neg_dataset) = (
            BuildMeanDecisionMap(result_idx, current_dataset[0].size(), current_dataset)
    );

    if (!info.pos_dataset.empty()) {
        std::tie(info.pos_homo, info.pos_r) = is_homogenous(info.pos_dataset, result_idx);
    } else {
        info.pos_homo = true;
        info.pos_r = 0.0; // Or some default value
    }

    if (!info.neg_dataset.empty()) {
        std::tie(info.neg_homo, info.neg_r) = is_homogenous(info.neg_dataset, result_idx);
    } else {
        info.neg_homo = true;
        info.neg_r = 0.0; // Or some default value
    }

    return info;
}

ai::DecisionTree::DecisionTree(data_set& T, size_t result_idx) : genesis_(nullptr) {
    SplitInfo info = collect_info(result_idx, T);

    genesis_ = new Node{nullptr, nullptr, info.bound, info.attr_idx, 0};
    if (!info.pos_dataset.empty()) {
        BuildBranch(info.pos_dataset, result_idx, genesis_, true);
    } else {
        genesis_->pos_child = new Node{nullptr, nullptr, 0, 0, info.pos_r}; // Leaf node
    }

    if (!info.neg_dataset.empty()) {
        BuildBranch(info.neg_dataset, result_idx, genesis_, false);
    } else {
        genesis_->neg_child = new Node{nullptr, nullptr, 0, 0, info.neg_r}; // Leaf node
    }
}

void ai::DecisionTree::BuildBranch(ai::data_set& current_dataset, size_t result_idx, Node*& branch_tail, bool link_pos) {
    SplitInfo info = collect_info(result_idx, current_dataset);
    auto* new_node = from_decision_info({info.attr_idx, info.bound, info.pos_dataset, info.neg_dataset});

    if (info.neg_homo || info.pos_homo) {
        if ((info.neg_homo or info.pos_dataset.empty())) {
            branch_tail->neg_child = new_node;
            new_node->result = info.neg_r;
        } else {
            branch_tail->neg_child = new ai::Node{nullptr, nullptr, 0, 0, 0};
        }

        if (info.pos_homo or info.neg_dataset.empty()) {
            branch_tail->pos_child = new_node;
            new_node->result = info.pos_r;
        } else {
            branch_tail->pos_child = new ai::Node{nullptr, nullptr, 0, 0, 0};
        }
        return;
    }

    if (link_pos)
        branch_tail->pos_child = new_node;
    else
        branch_tail->neg_child = new_node;

    if (!info.pos_dataset.empty()) {
        BuildBranch(info.pos_dataset, result_idx, new_node, true);
    } else {
        new_node->pos_child = new Node{nullptr, nullptr, 0, 0, info.pos_r}; // Leaf node
    }

    if (!info.neg_dataset.empty()) {
        BuildBranch(info.neg_dataset, result_idx, new_node, false);
    } else {
        new_node->neg_child = new Node{nullptr, nullptr, 0, 0, info.neg_r}; // Leaf node
    }
}
