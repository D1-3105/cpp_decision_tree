//
// Created by oleg on 15.07.24.
//

#include "CSVParser.h"


csv_parsing::CSVParser::CSVParser(std::string &input_file_path) : input_fp_(input_file_path) {
    reader_ =  std::make_shared<csv::CSVReader>(csv::CSVReader(input_fp_));
}

std::tuple<u_long, csv_parsing::databatch> csv_parsing::CSVParser::GetRows(u_long N) {
    csv_parsing::databatch rows(N);
    for (size_t i = 0; i < N; i++) {
        bool is_read = reader_->read_row(rows[i]);
        if (not is_read) {
            return {i, rows};
        }
    }
    return {N, rows};
}



