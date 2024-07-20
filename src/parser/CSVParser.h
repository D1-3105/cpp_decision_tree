//
// Created by oleg on 15.07.24.
//

#ifndef DECISION_TREE_CSVPARSER_H
#define DECISION_TREE_CSVPARSER_H

#include "iostream"
#include "csv.hpp"
#include "tuple"

namespace csv_parsing {

    using databatch = std::vector<csv::CSVRow>;
    using data = std::vector<std::vector<std::string&>>;

    class CSVParser {
    public:
        explicit CSVParser(std::string &input_file_path);
        std::tuple<u_long, std::vector<csv::CSVRow>> GetRows(u_long N);

    private:
        std::string& input_fp_;
        std::shared_ptr<csv::CSVReader> reader_;
    };
}




#endif //DECISION_TREE_CSVPARSER_H
