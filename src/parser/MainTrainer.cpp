//
// Created by oleg on 15.07.24.
//

#include <tbb/parallel_for.h>
#include "MainTrainer.h"

#include <mutex>

train::MainTrainer::MainTrainer(std::string &datasource_path) {
    parser_ = std::make_shared<csv_parsing::CSVParser>(csv_parsing::CSVParser(datasource_path));
    size_t _;
    std::vector<csv::CSVRow> out_row;
    std::tuple{_, out_row} = parser_->GetRows(1);
    vectorizer_ = std::make_shared<transform::Vectorizer>(transform::Vectorizer(out_row.size()));
}

ai::DecisionTree train::MainTrainer::Train(size_t train_size,
                               size_t result_attr_idx, const std::vector<std::function<float(const std::string &)>> processors) {
    ulong read_rows;
    csv_parsing::databatch batch;
    std::tie(read_rows, batch) = parser_->GetRows(train_size);
    ai::data_set data;
    std::mutex data_mutex; // Define a mutex to guard data
    tbb::parallel_for(
            tbb::blocked_range<size_t>(0, batch.size()),
            [=, this, &data, &data_mutex](tbb::blocked_range<size_t> pos) { // Capture data and data_mutex by reference
                for (size_t i = pos.begin(); i < pos.end(); i++) {
                    std::vector<std::string> row = batch[i];
                    auto vec = vectorizer_->Vectorize(row, const_cast<std::vector<std::function<float(const std::string &)>> &>(processors));
                    std::lock_guard<std::mutex> lock(data_mutex); // Lock the mutex before modifying data
                    data.push_back(vec);
                }
            });
    ai::DecisionTree tree(data, result_attr_idx);
    return tree;
}
