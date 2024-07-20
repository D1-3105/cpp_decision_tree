//
// Created by oleg on 15.07.24.
//

#ifndef DECISION_TREE_MAINTRAINER_H
#define DECISION_TREE_MAINTRAINER_H
#include "iostream"

#include "CSVParser.h"
#include "Vectorizer.h"
#include "DecisionTree.h"

namespace train {
    class MainTrainer {
    public:
        explicit MainTrainer(std::string& datasource_path);
        ai::DecisionTree Train(size_t train_size, size_t result_attr_idx, std::vector<std::function<float(const std::string &)>> processors);
        std::shared_ptr<csv_parsing::CSVParser> parser_;
        std::shared_ptr<transform::Vectorizer> vectorizer_;
    };
}



#endif //DECISION_TREE_MAINTRAINER_H
