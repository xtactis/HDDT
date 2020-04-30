#include <algorithm>
#include <iostream>
#include <fstream>
#include <numeric>
#include <vector>
#include <string>
#include <set>

#include <cstdio>
#include <cmath>

#include "profiler.hpp"

// TODO: dodati AUC i f-measure
// TODO: kad su vrijednosti continuous, podijeli u N buckets umjesto svih mogucih vrijednosti
// TODO: implementiraj multi-class hellinger distance
// TODO: dodaj commandline arguments
// TODO: dodaj cross validation
// TODO: fixati apsolutno sve da nije ovako fugly (npr don't mix iostream and stdio)
// TODO: nemoj koristiti std::vector nego napravi nesto svoje sto ce imati countove potrebne za hellingera
// TODO: dodaj grid search za hiperparametre
// TODO: dodaj mogucnost spremanja stvorenog stabla

const float TRAIN_TO_TEST_RATIO = 0.70f;
const int SEED = 69;
const int MAX_DEPTH = 5;
const bool HELLINGER = true;
const int BUCKETS = 10;


#define PROFILING 1
#if PROFILING
#define PROFILE_SCOPE(name) Timer timer##__LINE__(name)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)
#else
#define PROFILE_SCOPE(name)
#define PROFILE_FUNCTION()
#endif

union Z {
    int i;
    float f;
    Z() {}
    Z(int x): i(x) {}
    Z(float x): f(x) {}
};

std::vector<std::string> classes;
std::vector<std::string> attrNames;
std::vector<std::vector<std::string>> attrValues; // att[0] == {"b", "o", "x"}
std::vector<bool> isContinuous;

typedef std::vector<Z> Row;
typedef std::vector<Row> Rows;

namespace Utils {
    const double epsilon = 1e-6;
    bool eq(double x, double y) {
        return fabs(x-y) <= epsilon;
    }

    Rows filter(const Rows &rows, int label) {
        // filter only for rows with label == `label`
        Rows new_rows;
        for (const auto &row: rows) {
            if (row.back().i == label) {
                new_rows.push_back(row);
            }
        }
        return new_rows;
    }

    int count(const Rows &rows, int label) {
        PROFILE_FUNCTION();
        // count only for rows with label == `label`
        int ret = 0;
        for (const auto &row: rows) {
            if (row.back().i == label) {
                ++ret;
            }
        }
        return ret;
    }

    inline float sqr(float x) {
        return x*x;
    }

    void trim(std::string &s) {
        int start = 0, end = s.size()-1;
        while (s[start] == ' ') ++start;
        while (s[end] == ' ') --end;
        // "   abcdefgh   "
        //     ^      ^
        s = s.substr(start, end-start+1);
    }
}

struct Question {
    int column;
    float value; // TODO type pending // *mozda* nas nije briga

    bool match(const Row &example) const {
        // Compare the feature value in an example to the
        // feature value in this question.
        if (isContinuous[this->column]) {
            return example[this->column].f >= this->value;
        }
        return example[this->column].i == (int)this->value;
    }

    void print() const {
        printf("Is %s ", attrNames[this->column].c_str());
        if (isContinuous[this->column]) {
            printf(">= %.3f?\n", this->value);
            return;
        }
        printf("== %s?\n", attrValues[this->column][(int)this->value].c_str());
    }
};

std::vector<int> class_histogram(const Rows &rows) {
    /*
    count number of rows per class
    */
    std::vector<int> histogram(classes.size());
    for (const auto &row: rows) {
        int label = row.back().i;
        ++histogram[label];
    }
    return histogram;
}

struct Node {
    bool isLeaf = false;
    Node *left = nullptr, *right = nullptr;
    Question question;
    std::vector<int> predictions;

    Node(Node *l, Node *r, const Question &q) :
        left(l), right(r), question(q) {}

    Node(const Rows &rows) : isLeaf(true), predictions(class_histogram(rows)) {}

    static Node *DecisionNode(Node *l, Node *r, const Question &q) {
        return new Node(l, r, q);
    }

    static Node *Leaf(const Rows &rows) {
        return new Node(rows);
    }
};

float gini(const Rows &rows) {
    PROFILE_FUNCTION();
    /*
    Calculate the Gini Impurity for a list of rows.

    There are a few different ways to do this, I thought this one was
    the most concise. See:
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    */
    auto histogram = class_histogram(rows);
    float impurity = 1;
    for (int label: histogram) {
        float probability = 1. * label / rows.size();
        impurity -= probability*probability;
    }
    return impurity;
}

std::set<float> feature_values(int col, const Rows &rows) {
    // TODO: ne moze ostati float, ovisit ce o tipu attr
    // *mozda* nas to nije briga?
    std::set<float> values;
    for (const auto &row: rows) {
        values.insert(row[col].f);
    }
    return values;
}

void partition(const Rows &rows, const Question &question,
               Rows &true_rows, Rows &false_rows) {
    for (const auto &row: rows) {
        if (question.match(row)) {
            true_rows.push_back(row);
        } else {
            false_rows.push_back(row);
        }
    }
}

float info_gain(const Rows &left, const Rows &right,
                float current_uncertainty) {
    PROFILE_FUNCTION();
    float p = 1. * left.size() / (left.size() + right.size());
    return current_uncertainty - p*gini(left) - (1-p)*gini(right);
}

float hellinger_distance(const Rows &left, const Rows &right, float tp) {
    PROFILE_FUNCTION();
    float tfwp = Utils::count(right, 0);
    float tfvp = Utils::count(left, 0);
    float tfwn = right.size() - tfwp;
    float tfvn = left.size() - tfvp;
    float tn = (left.size() + right.size()) - tp;
    return Utils::sqr(std::sqrt(tfvp/tp) - std::sqrt(tfvn/tn))
         + Utils::sqr(std::sqrt(tfwp/tp) - std::sqrt(tfwn/tn));
}

void find_best_split(const Rows &rows, float &best_gain, 
                     Question &best_question) {
    PROFILE_FUNCTION();
    best_gain = 0;
    float current_uncertainty = gini(rows);
    float tp = Utils::count(rows, 0);
    int n_features = rows[0].size() - 1;
    for (int column = 0; column < n_features; ++column) {
        auto values = feature_values(column, rows);
        for (float value: values) {
            Question question = {column, value};
            Rows true_rows, false_rows;
            partition(rows, question, true_rows, false_rows);
            if (true_rows.size() == 0 || false_rows.size() == 0) {
                //std::cerr << "skip " << true_rows.size() << ' ' << false_rows.size() << std::endl;
                continue;
            }
            float gain;
            if (HELLINGER) {
                gain = hellinger_distance(true_rows, false_rows, tp);
            } else {
                gain = info_gain(true_rows, false_rows, current_uncertainty);
            }
            if (gain > best_gain) {
                best_gain = gain;
                best_question = question;
            }
        }
    }
    if (best_gain == 0) return;
    //best_question.print();
    //std::cerr << best_gain << std::endl;
}

Node *build_tree(const Rows &rows, int depth=MAX_DEPTH) {
    float gain;
    Question question;
    if (depth == 0) {
        return Node::Leaf(rows);
    }
    find_best_split(rows, gain, question);
    if (Utils::eq(gain, 0)) {
        return Node::Leaf(rows);
    }
    Rows true_rows, false_rows;
    partition(rows, question, true_rows, false_rows);

    Node *true_branch = build_tree(true_rows, depth-1);
    Node *false_branch = build_tree(false_rows, depth-1);
    return Node::DecisionNode(true_branch, false_branch, question);
}

void print_tree(Node *node, const std::string &spacing="") {
    if (node->isLeaf) {
        printf("%sPredict {", spacing.c_str());
        bool comma = false;
        for (int i = 0; i < (int)classes.size(); ++i) { // should always be 2
            if (node->predictions[i] == 0) continue;
            printf("%s%s: %d", (comma?", ":""), classes[i].c_str(), node->predictions[i]);
            comma = true;
        }
        printf("}\n");
        return;
    }
    printf("%s", spacing.c_str());
    node->question.print();

    printf("%s--> True:\n", spacing.c_str());
    print_tree(node->left, spacing+"  ");
    printf("%s--> False:\n", spacing.c_str());
    print_tree(node->right, spacing+"  ");
}

std::vector<int> classify(const Row &row, Node *node) {
    if (node->isLeaf) {
        return node->predictions;
    }
    if (node->question.match(row)) {
        return classify(row, node->left);
    }
    return classify(row, node->right);
}

float print_leaf(const std::vector<int> &counts, int actual) {
    float total = std::accumulate(counts.begin(), counts.end(), 0);
    //printf("{");
    bool comma = false;
    float ret = 0;
    for (int i = 0; i < (int)classes.size(); ++i) { // should always be 2
        float p = counts[i]/total*100;
        if (i == actual) ret = p;
        if (counts[i] == 0) continue;
        //printf("%s%s: %.2f%%", (comma?", ":""), classes[i].c_str(), p);
        comma = true;
    }
    //printf("}\n");
    return ret;
}

std::vector<std::string> parseLine(const std::string &line, char delimiter=',') {
    std::vector<std::string> splits;
    std::string cur;
    for (char c: line) {
        if (c == delimiter) {
            Utils::trim(cur);
            splits.push_back(cur);
            cur = "";
        } else {
            cur.push_back(c);
        }
    }
    Utils::trim(cur);
    if (!cur.empty()) {
        splits.push_back(cur);
    }
    return splits;
}

void makeDatastructure(const std::string &filePath) {
    PROFILE_FUNCTION();
    std::ifstream fin(filePath);
    std::string line;
    bool getClasses = true;
    bool getAttrs = false;
    for (int lineno = 0; getline(fin, line); ++lineno) {
        Utils::trim(line);
        if (line.size() == 0) continue;
        if (line.back() == '.') line.pop_back();
        if (line == "|classes") {
            getClasses = true;
        } else if (getClasses) {
            classes = parseLine(line);
            if (classes.size() == 1) {
                std::cerr << line << std::endl;
                std::cerr << classes[0] << std::endl;
                std::cerr << "What are you doing with just one class lol" << std::endl;
                exit(1);
            }
            getClasses = false;
            getAttrs = true;
        } else if (line == "|attributes") { // maybe unnecessary?
            getAttrs = true;
        } else if (getAttrs) {
            int colon = line.find(':');
            std::string attrName = line.substr(0, colon);
            attrNames.push_back(attrName);
            line = line.substr(colon+1);
            attrValues.push_back(parseLine(line));
            isContinuous.push_back(attrValues.back()[0] == "continuous");
        } else {
            std::cerr << "Unsupported .names file, check line " << lineno << std::endl;
            exit(1);
        }
    }
    /*for (int i = 0; i < (int)attrNames.size(); ++i) {
        std::cout << attrNames[i] << ": ";
        for (int j = 0; j < (int)attrValues[i].size(); ++j) {
            std::cout << attrValues[i][j] << ", ";
        }
        std::cout << std::endl;
    }*/
}

void getData(const std::string &filestub, 
             Rows &training_data, Rows &testing_data) {
    PROFILE_FUNCTION();
    makeDatastructure(filestub+".names");
    Rows data;
    std::ifstream fin(filestub+".data");
    std::string line;
    for (int lineno = 0; getline(fin, line); ++lineno) {
        auto values = parseLine(line);
        data.emplace_back(values.size());
        for (int i = 0; i < (int)values.size()-1; ++i) {
            if (values[i] == "?") {
                data.pop_back();
                break;
            }
            if (isContinuous[i]) {
                data.back()[i].f = std::atof(values[i].c_str());
            } else {
                auto x = std::find(attrValues[i].begin(),
                                   attrValues[i].end(),
                                   values[i]);
                if (x == attrValues[i].end()) {
                    std::cerr << "Looking for \"" << values[i] << "\" in:" << std::endl;
                    for (const auto &v: attrValues[i]) {
                        std::cerr << v << ", ";
                    }
                    std::cerr << "\nCheck line " << lineno << std::endl;
                    std::cerr << "That don't exist bro." << std::endl;
                    exit(1);
                }
                data.back()[i].i = x-attrValues[i].begin();
            }
        }
        auto x = std::find(classes.begin(),
                           classes.end(),
                           values.back());
        if (x == classes.end()) {
            std::cerr << "That class don't exist bro." << std::endl;
            exit(1);
        }
        data.back().back().i = x-classes.begin();
    }
    fin.close();

    /*for (const auto &row: data) {
        for (const auto &d: row) {
            printf("%f ", d.f);
        }
        puts("");
    }*/

    std::random_shuffle(data.begin(), data.end());
    int trainCount = data.size()*TRAIN_TO_TEST_RATIO;
    training_data.resize(trainCount);
    testing_data.resize(data.size()-trainCount);
    std::copy_n(data.begin(), trainCount, training_data.begin());
    std::copy_n(data.rbegin(), data.size()-trainCount, testing_data.begin());
}

int main(int argc, char **argv) {
    Instrumentor::get().beginSession("HDTV");
    PROFILE_FUNCTION();
    std::srand(SEED);
    Rows training_data, testing_data;
    std::string filestub; std::cin >> filestub; // e.g. "phoneme"
    getData(filestub, training_data, testing_data);

    Node *tree = build_tree(training_data);
    printf("\n");
    //print_tree(tree);
    float sum = 0;
    for (const auto &row: testing_data) {
        //printf("Actual: %s. Predicted: ", classes[row.back().i].c_str());
        sum += print_leaf(classify(row, tree), row.back().i);
    }
    printf("Average certainty: %.2f%%", sum/testing_data.size());
    Instrumentor::get().endSession();
}