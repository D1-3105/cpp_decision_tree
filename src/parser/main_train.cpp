#include "Vectorizer.h"
#include "map"

#include "MainTrainer.h"

const std::map<std::string, float> sex_map = {std::pair("F", -1.0), std::pair("M", 1.0)};
const std::map<std::string, float> drug_map = {
        std::pair("drugX", 10.0),
        std::pair("drugY", 20.0),
        std::pair("drugC", 30.0),
        std::pair("drugB", 40.0),
        std::pair("drugA", 50.0),
};

float transform_age(const std::string& age_str) {
    return atof(age_str.c_str());
}

float transform_sex(const std::string& sex_str) {
    auto founds = sex_map.find(sex_str);
    if (founds == sex_map.end()) {
        return 0.0;
    }
    return founds->second;
}

float transform_BP(const std::string& BP_str) {
    if (std::equal(BP_str.begin(), BP_str.end(), "HIGH")) {
        return 1.0;
    } else if (std::equal(BP_str.begin(), BP_str.end(), "LOW")) {
        return -1.0;
    } else {
        return 0.0;
    }
}

float transform_Cholesterol(const std::string& Cholesterol_str) {
    if (std::equal(Cholesterol_str.begin(), Cholesterol_str.end(), "HIGH")) {
        return 1.0;
    } else if (std::equal(Cholesterol_str.begin(), Cholesterol_str.end(), "LOW")) {
        return -1.0;
    } else {
        return 0.0;
    }
}

float transform_Na_to_K(const std::string& Na_to_K_str) {
    return atof(Na_to_K_str.c_str());
}

float transform_Drug(const std::string& Drug_str) {
    auto founds = drug_map.find(Drug_str);
    if (founds == drug_map.end()) {
        return 0.0;
    }
    return founds->second;
}


int main() {
    std::string data_source = getenv("DATASOURCE");

    std::vector<std::function<float(const std::string&)>> processors = {
            transform_age, transform_sex, transform_BP, transform_Cholesterol, transform_Na_to_K, transform_Drug
    };
    train::MainTrainer trainer(data_source);
    auto tree = trainer.Train(180, 5, processors);
    bool _;
    csv_parsing::databatch d;
    std::tie(_, d) = trainer.parser_->GetRows(1);
    auto vectorized = trainer.vectorizer_->Vectorize(d[0], processors);
    auto res = tree.Predict(vectorized);
    std::cout << "Expected: " << vectorized[5] << " Predicted: " << res << std::endl;
    return 0;
}
