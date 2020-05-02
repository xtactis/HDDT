#include <algorithm>
#include <fstream>
#include <numeric>
#include <vector>
#include <string>
#include <set>
#include <map>

#include <cstdio>
#include <cmath>
#include <cstring>

#include "profiler.hpp"

// TODO: implementiraj multi-class hellinger distance
// TODO: sve osim stats u log file or sth
// TODO: dodaj cross validation
// TODO: fixati apsolutno sve da nije ovako fugly (npr don't mix iostream and stdio)
// TODO: nemoj koristiti std::vector nego napravi nesto svoje sto ce imati countove potrebne za hellingera
// TODO: dodaj grid search za hiperparametre
// TODO: dodaj mogucnost spremanja stvorenog stabla

//#define PROFILING
#ifdef PROFILING
#define PROFILE_SCOPE(name) Timer timer##__LINE__(name);
#define PROFILE_FUNCTION PROFILE_SCOPE(__FUNCTION__)
#else
#define PROFILE_SCOPE(name)
#define PROFILE_FUNCTION
#endif

union Z {
    int i;
    float f;
    Z() {}
    Z(int x): i(x) {}
    Z(float x): f(x) {}
};

using Row = std::vector<Z>;
using Rows = std::vector<Row>;

int minorityClass; // TODO: get rid of these globals, legitimately just send them to everything that uses them
std::vector<int> totalHist;
std::vector<std::string> classes;
std::vector<std::string> attrNames;
std::vector<std::vector<std::string>> attrValues; // att[0] == {"b", "o", "x"}
std::vector<bool> isContinuous;
std::vector<float> mins, maxs;
std::vector<int> uniqueValueCount;

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
    Z value; // TODO type pending // *mozda* nas nije briga

    bool match(const Row &example) const {
        // Compare the feature value in an example to the
        // feature value in this question.
        if (isContinuous[this->column]) {
            return example[this->column].f >= this->value.f;
        }
        return example[this->column].i == (int)this->value.i;
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
        if (isContinuous[col])
            values.insert(row[col].f);
        else
            values.insert(row[col].i);
    }
    return values;
}

// u rows koliko ima labela == minorityClass takvih da je question.match(row)

void partition(const Rows &rows, const Question &question, Rows &true_rows, Rows &false_rows) {
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
    float p = 1. * left.size() / (left.size() + right.size());
    return current_uncertainty - p*gini(left) - (1-p)*gini(right);
}

float hellinger_distance(int lsize, int rsize, float tp, float tfvp, float tfwp) {
    float tfvn = lsize - tfvp;
    float tfwn = rsize - tfwp;
    float tn = (lsize + rsize) - tp;
    return Utils::sqr(std::sqrt(tfvp/tp) - std::sqrt(tfvn/tn))
         + Utils::sqr(std::sqrt(tfwp/tp) - std::sqrt(tfwn/tn));
}

void split_continuous(const Rows &rows, int column, float current_uncertainty,
                                float &best_gain, Question &best_question,
                                int max_buckets) {
    float min = mins[column], max = maxs[column];
    const int buckets = std::min(max_buckets, uniqueValueCount[column]);
    float step = (max-min)/buckets;
    for (int bucket = 0; bucket <= buckets; ++bucket) {
        float value = bucket*step+min;
        Question question = {column, value};
        Rows true_rows, false_rows;
        partition(rows, question, true_rows, false_rows);
        if (true_rows.size() == 0 || false_rows.size() == 0) {
            continue;
        }
        float gain = info_gain(true_rows, false_rows, current_uncertainty);
        if (gain > best_gain) {
            best_gain = gain;
            best_question = question;
        }
    }
}

void split_discrete(const Rows &rows, int column, float current_uncertainty,
                               float &best_gain, Question &best_question) {
    for (int value = 0; value < (int)attrValues[column].size(); ++value) {
        Question question = {column, value};
        Rows true_rows, false_rows;
        partition(rows, question, true_rows, false_rows);
        if (true_rows.size() == 0 || false_rows.size() == 0) {
            continue;
        }
        float gain = info_gain(true_rows, false_rows, current_uncertainty);
        if (gain > best_gain) {
            best_gain = gain;
            best_question = question;
        }
    }
}

void find_best_split_indian(const Rows &rows, float &best_gain, 
                            Question &best_question, int max_buckets) {
    best_gain = 0;
    float current_uncertainty = gini(rows);
    int n_features = rows[0].size() - 1;
    for (int column = 0; column < n_features; ++column) {
        if (isContinuous[column]) {
            split_continuous(rows, column, current_uncertainty, best_gain, best_question, max_buckets);
        } else {
            split_discrete(rows, column, current_uncertainty, best_gain, best_question);
        }
    }
}

void hellinger_split_continuous(const Rows &rows, int column, int tp,
                                float &best_gain, Question &best_question,
                                int max_buckets) {
    float min = mins[column], max = maxs[column];
    const int buckets = std::min(max_buckets, uniqueValueCount[column]);
    float step = (max-min)/buckets;
    for (int bucket = 0; bucket <= buckets; ++bucket) {
        float value = step*bucket+min;
        Question question = {column, value};
        int lsize = 0, rsize = 0;
        int tfvp = 0, tfwp = 0;
        for (const auto &row: rows) {
            if (question.match(row)) {
                ++lsize;
                if (row.back().i == minorityClass) ++tfvp;
            } else {
                ++rsize;
                if (row.back().i == minorityClass) ++tfwp;
            }
        }
        if (lsize == 0 || rsize == 0) {
            continue;
        }
        float gain = hellinger_distance(lsize, rsize, tp, tfvp, tfwp);
        if (gain > best_gain) {
            best_gain = gain;
            best_question = question;
        }
    }
}

void hellinger_split_discrete(const Rows &rows, int column, int tp,
                               float &best_gain, Question &best_question) {
    for (int value = 0; value < (int)attrValues[column].size(); ++value) {
        Question question = {column, value};
        int lsize = 0, rsize = 0;
        int tfvp = 0, tfwp = 0;
        for (const auto &row: rows) {
            if (question.match(row)) {
                ++lsize;
                if (row.back().i == minorityClass) ++tfvp;
            } else {
                ++rsize;
                if (row.back().i == minorityClass) ++tfwp;
            }
        }
        if (lsize == 0 || rsize == 0) {
            continue;
        }
        float gain = hellinger_distance(lsize, rsize, tp, tfvp, tfwp);
        if (gain > best_gain) {
            best_gain = gain;
            best_question = question;
        }
    }
}

void find_best_split_hellinger(const Rows &rows,
                               float &best_gain, Question &best_question,
                               int max_buckets) {
    best_gain = 0;
    float tp = Utils::count(rows, minorityClass);
    int n_features = rows[0].size() - 1;
    for (int column = 0; column < n_features; ++column) {
        if (isContinuous[column]) {
            hellinger_split_continuous(rows, column, tp, best_gain, best_question, max_buckets);
        } else {
            hellinger_split_discrete(rows, column, tp, best_gain, best_question);
        }
    }
}

Node *build_hellinger_tree(const Rows &rows, int depth, int max_buckets) {
    float gain; 
    Question question;
    if (depth == 0) {
        return Node::Leaf(rows);
    }
    // nez
    find_best_split_hellinger(rows, gain, question, max_buckets);
    if (Utils::eq(gain, 0)) {
        return Node::Leaf(rows);
    }
    Rows true_rows, false_rows;
    partition(rows, question, true_rows, false_rows);

    Node *true_branch = build_hellinger_tree(true_rows, depth-1, max_buckets);
    Node *false_branch = build_hellinger_tree(false_rows, depth-1, max_buckets);
    return Node::DecisionNode(true_branch, false_branch, question);
}

Node *build_decision_tree(const Rows &rows, int depth, int max_buckets) {
    float gain; 
    Question question;
    if (depth == 0) {
        return Node::Leaf(rows);
    }
    find_best_split_indian(rows, gain, question, max_buckets);
    if (Utils::eq(gain, 0)) {
        return Node::Leaf(rows);
    }
    Rows true_rows, false_rows;
    partition(rows, question, true_rows, false_rows);

    Node *true_branch = build_decision_tree(true_rows, depth-1, max_buckets);
    Node *false_branch = build_decision_tree(false_rows, depth-1, max_buckets);
    return Node::DecisionNode(true_branch, false_branch, question);
}

/* TODO: finish later
struct NekaStrukturica {
    //lol[5][3]; // za atribut 5 cija je vrijednost >= BUCKET[0.3] koliko ima minority klasa?
    std::vector<int> count;
    Z min, max;
};

auto nez(Rows &rows) {
    std::vector<NekaStrukturica> lol(rows[0].size()-1);
    for (int column = 0; column < (int)rows[0].size()-1; ++column) {
        sort(rows.begin(), rows.end(), [column](const Row &a, const Row &b){
            if (isContinuous[column]) {
                return a[column].f < b[column].f;
            }
            return a[column].i < b[column].i;
        });
        if (isContinuous[column]) {
            lol[column].min = rows[0][column];
            lol[column].max = rows.back()[column];
            lol[column].count.resize(BUCKETS);
            fprintf(stderr, "\ncol: %d; min: %.2f; max: %.2f;\n", column, lol[column].min.f, lol[column].max.f);
        } else {
            lol[column].min = rows[0][column];
            lol[column].max = rows.back()[column];
            lol[column].count.resize(lol[column].max.i - lol[column].min.i + 1);
            fprintf(stderr, "col: %d; min: %d; max: %d; size: %d\n", column, lol[column].min.i, lol[column].max.i, lol[column].count.size());
        }
        float prevf = 0;
        if (isContinuous[column]) {
            for (int bucket = 0; bucket < BUCKETS; ++bucket) {
                float f = rows[row][column].f-prevf;
                float step = (lol[column].max.f-lol[column].min.f)/BUCKETS;
                if (row == 0 || f > step) {
                    prevf = f;
                    lol[column].count[bucket++] = rows.size()-row;
                    fprintf(stderr, "row: %d; bucket: %d (%.2f); count: %d;\n", row, bucket-1, prevf, lol[column].count[bucket-1]);
                }
            }
        } else {
            for (int row = 0, bucket = 0; row < (int)rows.size(); ++row) {
                if (row == 0 || (rows[row][column].i != rows[column][row-1].i)) {
                    lol[column].count[rows[row][column].i] = rows.size()-row;
                    fprintf(stderr, "row: %d; index: %d; count: %d;\n", row, rows[row][column].i, lol[column].count[rows[row][column].i]);
                }
            }
        }
    }
    return lol;
}
*/

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
    printf("Is %s ", attrNames[node->question.column].c_str());
    if (isContinuous[node->question.column]) {
        printf(">= %.3f?\n", node->question.value.f);
        return;
    }
    printf("== %s?\n", attrValues[node->question.column][(int)node->question.value.i].c_str());

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

void print_leaf(const std::vector<int> &counts, int total) {
    printf("{");
    bool comma = false;
    for (int i = 0; i < (int)classes.size(); ++i) { // should always be 2
        float p = 1.*counts[i]/total*100;
        if (counts[i] == 0) continue;
        printf("%s%s: %.2f%%", (comma?", ":""), classes[i].c_str(), p);
        comma = true;
    }
    printf("}\n");
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
                fprintf(stderr, "%s\n%s\nWhat are you doing with just one class lol\n", line.c_str(), classes[0].c_str());
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
            fprintf(stderr, "Unsupported .names file, check line %d\n", lineno);
            fin.close();
            exit(1);
        }
    }
    fin.close();
}

void getData(const std::string &filestub, Rows &data) {
    makeDatastructure(filestub+".names");
    std::ifstream fin(filestub+".data");
    std::string line;
    mins.resize(attrValues.size(), NAN);
    maxs.resize(attrValues.size(), NAN);
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
                if (std::isnan(mins[i])) {
                    mins[i] = data.back()[i].f;
                    maxs[i] = data.back()[i].f;
                }
                mins[i] = std::min(mins[i], data.back()[i].f);
                maxs[i] = std::max(maxs[i], data.back()[i].f);
            } else {
                auto x = std::find(attrValues[i].begin(),
                                   attrValues[i].end(),
                                   values[i]);
                if (x == attrValues[i].end()) {
                    fprintf(stderr, "Looking for \"%s\" in:\n", values[i].c_str());
                    for (const auto &v: attrValues[i]) {
                        fprintf(stderr, "%s, ", v.c_str());
                    }
                    fprintf(stderr, "\nCheck line %d\nThat don't exist bro.\n", lineno);
                    exit(1);
                }
                data.back()[i].i = x-attrValues[i].begin();
            }
        }
        auto x = std::find(classes.begin(),
                           classes.end(),
                           values.back());
        if (x == classes.end()) {
            fprintf(stderr, "That class don't exist bro.");
            exit(1);
        }
        data.back().back().i = x-classes.begin();
    }
    fin.close();

    uniqueValueCount.resize(data[0].size()-1);
    for (int column = 0; column < (int)data[0].size()-1; ++column) {
        uniqueValueCount[column] = feature_values(column, data).size();
    }

    if (classes.size() == 2) { // TODO: bilo bi lipo da radi i za vise od 2 klasice
        totalHist.resize(2);
        totalHist[0] = Utils::count(data, 0);
        totalHist[1] = data.size()-totalHist[0];
    }
    minorityClass = min_element(totalHist.begin(), totalHist.end())-totalHist.begin();
}

float AUC(int TP, int TN, int FP, int FN) {
    return (1.f*TP/(TP+FN) + 1.f*TN/(FP+TN))/2.f;
}

float fMeasure(int TP, int , int FP, int FN) {
    float precision = 1.*TP/(TP+FP);
    float sensitivity = 1.*TP/(TP+FN);
    return 2.*precision*sensitivity/(precision+sensitivity);
}

int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "-h") == 0) {
        fprintf(stderr, "-h\tTo display this help message\n");
        fprintf(stderr, "-s\tTo set the seed, 69 by default\n");
        fprintf(stderr, "-HD\tuse Hellinger distance\n");
        fprintf(stderr, "-f\tset filestem, expects <filestem>.data and <filestem>.names\n");
        // fprintf(stderr, "-t\tuse separate test file, expected <filestem>.test\n"); later
        fprintf(stderr, "-T\tset train to test ratio if no separate test file, 0.7 by default\n");
        fprintf(stderr, "-b\tset max buckets for continuous values, 10 by default\n");
        fprintf(stderr, "-d\tset max depth, 999 by default\n");
        // fprintf(stderr, "-c\tset cross validation\n"); later
        return 0;
    }
    int seed = 69;
    bool hellinger = false;
    std::string filestem; // ew std::string
    float train_to_test_ratio = 0.7f;
    int max_buckets = 10;
    int max_depth = 999;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-s") == 0) {
            if (++i < argc) {
                seed = std::atoi(argv[i]);
            } else {
                fprintf(stderr, "-s requires a value");
                return 1;
            }
        } else if (strcmp(argv[i], "-HD") == 0) {
            hellinger = true;
        } else if (strcmp(argv[i], "-f") == 0) {
            if (++i < argc) {
                filestem = argv[i];
            } else {
                fprintf(stderr, "-f requires a value");
                return 1;
            }
        } else if (strcmp(argv[i], "-T") == 0) {
            if (++i < argc) {
                train_to_test_ratio = std::atof(argv[i]);
            } else {
                fprintf(stderr, "-T requires a value");
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0) {
            if (++i < argc) {
                max_buckets = std::atoi(argv[i]);
            } else {
                fprintf(stderr, "-b requires a value");
                return 1;
            }
        } else if (strcmp(argv[i], "-d") == 0) {
            if (++i < argc) {
                max_depth = std::atoi(argv[i]);
            } else {
                fprintf(stderr, "-d requires a value");
                return 1;
            }
        }
    }

    Instrumentor::get().beginSession("HDTV");
    std::srand(seed);
    Rows data;
    getData(filestem, data);

    Rows training_data, testing_data;
    std::random_shuffle(data.begin(), data.end());
    int trainCount = data.size()*train_to_test_ratio;
    training_data.resize(trainCount);
    testing_data.resize(data.size()-trainCount);
    std::copy_n(data.begin(), trainCount, training_data.begin());
    std::copy_n(data.rbegin(), data.size()-trainCount, testing_data.begin());
    
    //auto nn = nez(training_data);
    std::random_shuffle(training_data.begin(), training_data.end()); // unsort

    Node *tree;
    if (hellinger) {
        fprintf(stderr, "Building Hellinger tree...\n");
        tree = build_hellinger_tree(training_data, max_depth, max_buckets);
    } else {
        fprintf(stderr, "Building decision tree...\n");
        tree = build_decision_tree(training_data, max_depth, max_buckets);
    }
    printf("\n");
    //print_tree(tree);
    float sum = 0;
    int TP = 0, TN = 0, FP = 0, FN = 0;
    int truers = 0;
    for (const auto &row: testing_data) {
        //printf("Actual: %s. Predicted: ", classes[row.back().i].c_str());
        int actual = row.back().i;
        const auto hist = classify(row, tree);
        float total = std::accumulate(hist.begin(), hist.end(), 0);
        //print_leaf(hist, total);
        int prediction = std::max_element(hist.begin(), hist.end())-hist.begin();
        sum += hist[actual]/total;
        if (totalHist.size() == 2) {
            if (prediction == actual) {
                ++truers;
                if (prediction == minorityClass) ++TP;
                else ++TN;
            } else {
                if (prediction == minorityClass) ++FP;
                else ++FN;
            }
        }
    }
    printf("Average confidence:\t%8.5f%%\n", sum/testing_data.size()*100);
    printf("Accuracy:\t\t%8.5f%%\n", 1.*truers/testing_data.size()*100);
    if (totalHist.size() == 2) {
        printf("AUC:\t\t\t%8.5f\n", AUC(TP, TN, FP, FN));
        printf("f-measure:\t\t%8.5f", fMeasure(TP, TN, FP, FN));
    }
    Instrumentor::get().endSession();
    return 0;
}